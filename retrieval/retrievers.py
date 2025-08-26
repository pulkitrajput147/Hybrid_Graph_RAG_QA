# 2_retrieval/retrievers.py

from typing import List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.indices.property_graph import PropertyGraphIndex  # Updated import
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
import logging

from core.config import get_vector_store, get_graph_store, Settings

logger = logging.getLogger(__name__)

class HybridGraphRetriever(BaseRetriever):
    """
    A custom hybrid retriever that implements the Vector-First, Graph-Expansion pattern.
    1. Performs a vector search to get initial context.
    2. Extracts entities from the retrieved text.
    3. Traverses the knowledge graph from those entities to enrich the context.
    """
    def __init__(self, vector_index: VectorStoreIndex, graph_index: PropertyGraphIndex, top_k: int = 10):
        self._vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        self._graph_retriever = graph_index.as_retriever(include_text=True)  # Include text for better context
        self._llm = Settings.llm
        self.top_k = top_k
        super().__init__()

    def _get_entities_from_text(self, text: str) -> List[str]:
        """Uses an LLM to extract key entities from a text block."""

        logger.info("The received text is :\n",{text})
        logger.info("\n")
        prompt = PromptTemplate(
            "Based on the following text about government budget documents, identify and list the key entities "
            "(e.g., departments, budget items, locations, monetary figures, years, programs). "
            "Focus on specific, searchable terms that could be nodes in a knowledge graph. "
            "Output them as a comma-separated list with no additional text.\n\n"
            "Text: {text}\n\n"
            "Entities:"
        )
        try:
            response = self._llm.complete(prompt.format(text=text))
            entities = [entity.strip() for entity in response.text.split(",") if entity.strip()]
            # Filter out very short or generic entities
            return [e for e in entities if len(e) > 2 and not e.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at']]
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.info("=== Hybrid Graph Retrieval Started ===")
        
        # 1. Vector Search
        logger.info("Step 1: Performing vector search...")
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        logger.info("Vector nodes are :\n",vector_nodes)
        logger.info(f"Vector search returned {len(vector_nodes)} nodes")
        
        # 2. Entity Linking
        logger.info("Step 2: Extracting entities from vector search results...")
        all_entities = set()
        for node in vector_nodes:
            entities = self._get_entities_from_text(node.get_content())
            all_entities.update(entities)
        
        logger.info(f"Found entities: {list(all_entities)[:10]}...")  # Show first 10
        
        # 3. Graph Traversal
        retrieved_nodes = list(vector_nodes)
        if all_entities:
            logger.info("Step 3: Expanding with graph traversal...")
            try:
                # Create multiple focused queries from entities
                entity_batches = list(all_entities)[:10]  # Limit to prevent overwhelming
                for entity in entity_batches:
                    graph_query = f"{query_bundle.query_str} {entity}"
                    graph_nodes = self._graph_retriever.retrieve(QueryBundle(query_str=graph_query))
                    
                    # Add unique nodes
                    existing_node_ids = {node.node_id for node in retrieved_nodes}
                    for node in graph_nodes:
                        if node.node_id not in existing_node_ids:
                            retrieved_nodes.append(node)
                            existing_node_ids.add(node.node_id)
                            
            except Exception as e:
                logger.warning(f"Graph traversal failed: {e}")
        
        logger.info(f"Total retrieved nodes: {len(retrieved_nodes)}")
        logger.info(f"retrieved nodes\n: {(retrieved_nodes)}")
        return retrieved_nodes[:self.top_k * 4]  # Return reasonable number of nodes