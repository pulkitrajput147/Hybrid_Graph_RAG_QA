# 2_retrieval/engine.py
import logging
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph.transformations import SimpleLLMPathExtractor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.core.query_engine.sub_question_query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.prompts import PromptTemplate
from llama_index.core import QueryBundle
from llama_index.core.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import DecomposeQueryTransform

from core.config import get_vector_store, get_graph_store, Settings
from .retrievers import HybridGraphRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Advanced prompt for reasoned synthesis
SYNTHESIS_PROMPT_TEMPLATE = """
You are an expert financial analyst AI. Your task is to answer questions about multi-year government budget documents.
You must answer the user's QUESTION using ONLY the information provided in the CONTEXT section.

The CONTEXT contains two types of information:
1. Unstructured text chunks from the original budget documents.
2. Structured Knowledge Graph data in the form of (subject, predicate, object) triples.

To answer the user's comparative question, you must follow these reasoning steps:
1. From the CONTEXT, identify the relevant facts, figures, and narratives for each year or item mentioned in the QUESTION.
2. Use the structured graph data for specific, hard facts like budget allocations.
3. Use the unstructured text for explanatory details, commentary, and rationale.
4. If a calculation is needed (e.g., percentage change), perform it based on the retrieved facts.
5. Synthesize your findings into a clear, concise final answer.
6. For every fact you state, you MUST include a citation back to the source document (e.g., [budget_2024.pdf]). The source is available in the metadata of the context.

Let's think step by step.

CONTEXT:
---------------------
{context_str}
---------------------
QUESTION: {query_str}

FINAL ANSWER:
"""

def create_base_query_engine():
    """Creates the base hybrid graph query engine without decomposition."""
    # Initialize components
    vector_store = get_vector_store()
    graph_store = get_graph_store()
    
    # Create indices
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        kg_extractors=[
            SimpleLLMPathExtractor(llm=Settings.llm)
        ],
        embed_kg_nodes=True, 
    )

    # Setup the Hybrid Retriever
    hybrid_retriever = HybridGraphRetriever(vector_index, graph_index)

    # Setup the Synthesizer with the advanced CoT prompt
    synthesis_prompt = PromptTemplate(SYNTHESIS_PROMPT_TEMPLATE)
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", 
        summary_template=synthesis_prompt,
    )
    
    # Create the base query engine with the hybrid retriever
    base_query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
    )
    
    return base_query_engine

def create_multistep_query_engine():
    """
    Creates a MultiStepQueryEngine that decomposes complex queries into sequential steps.
    This is the modern replacement for StepDecomposeQueryTransform.
    """
    base_engine = create_base_query_engine()
    
    # Create MultiStepQueryEngine - this handles query decomposition internally
    multistep_engine = MultiStepQueryEngine(
        query_engine=base_engine,
        query_transform=DecomposeQueryTransform(llm=Settings.llm, verbose=True),
        index_summary="A hybrid graph RAG system for government budget document analysis that combines vector search with knowledge graph traversal.",
        num_steps=3,  # Maximum number of decomposition steps
    )
    
    logger.info("✅ MultiStep Query Engine created successfully.")
    return multistep_engine

def create_subquestion_query_engine():
    """
    Alternative approach using SubQuestionQueryEngine for complex queries.
    This breaks down complex queries into parallel sub-questions.
    """
    base_engine = create_base_query_engine()
    
    # Create query engine tools for different aspects of budget analysis
    query_engine_tools = [
        QueryEngineTool(
            query_engine=base_engine,
            metadata=ToolMetadata(
                name="budget_analyzer",
                description="Analyze government budget documents including financial data, departmental allocations, and year-over-year comparisons"
            ),
        )
    ]
    
    # Create SubQuestionQueryEngine
    subquestion_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=None,  # Will use global service context
        use_async=True,
        verbose=True,
    )
    
    logger.info("✅ SubQuestion Query Engine created successfully.")
    return subquestion_engine

def create_query_engine(decomposition_type="none"):
    """
    Main factory function to create different types of query engines.
    
    Args:
        decomposition_type (str): "none", "multistep", or "subquestion"
    """
    if decomposition_type == "multistep":
        return create_multistep_query_engine()
    elif decomposition_type == "subquestion":
        return create_subquestion_query_engine()
    else:
        base_engine = create_base_query_engine()
        logger.info("✅ Base Hybrid Graph RAG Query Engine created successfully.")
        return base_engine

# Custom Query Decomposer (if you want more control)
class CustomQueryDecomposer:
    """
    Custom query decomposer that understands budget document queries better.
    """
    def __init__(self, llm):
        self.llm = llm
        self.decomposition_prompt = PromptTemplate(
            "You are a query decomposition expert for government budget analysis. "
            "Break down the following complex query into 2-4 simpler, sequential sub-queries "
            "that can be answered step by step. Each sub-query should focus on a specific aspect "
            "like: identifying entities, finding specific data, making comparisons, or calculations.\n\n"
            "Original Query: {query}\n\n"
            "Decomposed Sub-queries (one per line):\n"
        )
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries."""
        response = self.llm.complete(self.decomposition_prompt.format(query=query))
        sub_queries = [q.strip() for q in response.text.split('\n') if q.strip()]
        return sub_queries[:4]  # Limit to 4 sub-queries

def create_custom_decomposition_engine():
    """
    Creates a query engine with custom query decomposition logic.
    """
    base_engine = create_base_query_engine()
    decomposer = CustomQueryDecomposer(Settings.llm)
    
    class CustomDecompositionEngine:
        def __init__(self, base_engine, decomposer):
            self.base_engine = base_engine
            self.decomposer = decomposer
        
        def query(self, query_str: str):
            """Query with custom decomposition."""
            logger.info(f"Processing complex query: {query_str}")
            
            # Decompose the query
            sub_queries = self.decomposer.decompose_query(query_str)
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            
            # Process each sub-query and collect responses
            responses = []
            context_accumulator = ""
            
            for i, sub_query in enumerate(sub_queries):
                logger.info(f"Processing sub-query {i+1}: {sub_query}")
                
                # Add previous context to current query
                enhanced_query = f"{sub_query}\n\nPrevious context: {context_accumulator}"
                response = self.base_engine.query(QueryBundle(query_str=enhanced_query))
                responses.append(response)
                context_accumulator += f"\n{response.response}"
            
            # Synthesize final response
            final_query = f"Based on the following sub-query responses, provide a comprehensive answer to: {query_str}\n\n"
            for i, resp in enumerate(responses):
                final_query += f"Sub-query {i+1} response: {resp.response}\n\n"
            
            final_response = self.base_engine.query(QueryBundle(query_str=final_query))
            return final_response
    
    custom_engine = CustomDecompositionEngine(base_engine, decomposer)
    logger.info("✅ Custom Decomposition Query Engine created successfully.")
    return custom_engine