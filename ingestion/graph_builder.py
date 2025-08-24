# 1_ingestion/graph_builder.py

from typing import List, Dict, Any, Literal
from pathlib import Path
import logging
import json

from llama_index.core import Document, StorageContext
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph.transformations import (
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
    DynamicLLMPathExtractor
)
from llama_index.core.graph_stores.types import EntityNode, Relation

from core.config import get_graph_store, get_neo4j_driver, Settings

logger = logging.getLogger(__name__)

def clean_document_metadata_for_graph(documents: List[Document]) -> List[Document]:
    """Clean and severely limit metadata size for graph processing"""
    cleaned_docs = []
    
    for doc in documents:
        try:
            # Keep only absolutely essential metadata
            essential_metadata = {}
            
            # Only keep these critical fields
            critical_fields = ['file_name', 'document_year', 'chunk_id']
            
            for key in critical_fields:
                if key in doc.metadata:
                    value = doc.metadata[key]
                    if isinstance(value, str):
                        essential_metadata[key] = value[:30]  # Very short strings
                    elif isinstance(value, (int, float)):
                        essential_metadata[key] = value
            
            # Create clean document
            content = doc.get_content()
            if len(content) > 3000:  # Limit content size for graph processing
                content = content[:3000] + "..."
            
            cleaned_doc = Document(
                text=content,
                metadata=essential_metadata
            )
            cleaned_docs.append(cleaned_doc)
            
        except Exception as e:
            logger.warning(f"Error cleaning document for graph: {e}")
            # Create minimal document
            cleaned_doc = Document(
                text=doc.get_content()[:1000],
                metadata={'file_name': 'unknown', 'chunk_id': len(cleaned_docs)}
            )
            cleaned_docs.append(cleaned_doc)
    
    return cleaned_docs

# Simplified schema for budget documents
class BudgetSchema:
    """Simplified schema definition for budget document knowledge graph"""
    
    # Entity types - simplified
    entities = [
        "BUDGET_DOCUMENT",
        "FISCAL_YEAR", 
        "DEPARTMENT",
        "PROGRAM",
        "AMOUNT",
        "PROJECT"
    ]
    
    # Relationship types - simplified
    relations = [
        "BELONGS_TO",
        "ALLOCATED_TO",
        "CONTAINS",
        "FUNDS",
        "PART_OF"
    ]

class EnhancedGraphBuilder:
    """Simplified graph builder with better error handling"""
    
    def __init__(self):
        self.graph_store = get_graph_store()
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
    def build_graph_with_multiple_extractors(self, documents: List[Document]) -> PropertyGraphIndex:
        """Build graph with simplified approach and better error handling"""
        
        logger.info("Building knowledge graph with simplified extractors...")

        # Clean metadata aggressively to prevent issues
        documents = clean_document_metadata_for_graph(documents)
        
        # Limit number of documents for graph processing
        if len(documents) > 50:
            logger.info(f"Limiting documents for graph processing: {len(documents)} -> 50")
            documents = documents[:50]
        
        # Initialize extractors with error handling
        extractors = self._initialize_extractors()
        
        if not extractors:
            raise ValueError("No extractors could be initialized")
        
        logger.info(f"Using {len(extractors)} extractors for graph construction")
        
        # Create PropertyGraphIndex with simplified approach
        try:
            index = self._create_graph_index_safe(documents, extractors)
            logger.info("✅ Base knowledge graph construction completed.")
            
            # Add basic enhancements
            self._add_basic_enhancements(documents)
            
            return index
            
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
            raise
    
    def _initialize_extractors(self):
        """Initialize extractors with fallback options"""
        extractors = []
        
        # Try Simple extractor first (most reliable)
        try:
            simple_extractor = SimpleLLMPathExtractor(llm=Settings.llm)
            extractors.append(simple_extractor)
            logger.info("✅ Simple extractor initialized")
        except Exception as e:
            logger.warning(f"Simple extractor failed: {e}")
        
        # Try Schema extractor with minimal parameters
        try:
            schema_extractor = SchemaLLMPathExtractor(
                llm=Settings.llm,
                possible_entities=BudgetSchema.entities,
                possible_relations=BudgetSchema.relations,
                strict=False
            )
            extractors.append(schema_extractor)
            logger.info("✅ Schema extractor initialized")
        except Exception as e:
            logger.warning(f"Schema extractor failed: {e}")
        
        # Try Dynamic extractor with conservative settings
        try:
            dynamic_extractor = DynamicLLMPathExtractor(
                llm=Settings.llm,
                max_triplets_per_chunk=3  # Very conservative
            )
            extractors.append(dynamic_extractor)
            logger.info("✅ Dynamic extractor initialized")
        except Exception as e:
            logger.warning(f"Dynamic extractor failed: {e}")
        
        return extractors
    
    def _create_graph_index_safe(self, documents: List[Document], extractors: List) -> PropertyGraphIndex:
        """Create graph index with progressive fallback"""
        
        # Try with all extractors first
        try:
            logger.info("Trying with all available extractors...")
            index = PropertyGraphIndex.from_documents(
                documents,
                kg_extractors=extractors,
                storage_context=self.storage_context,
                embed_kg_nodes=True,
                show_progress=True,
            )
            return index
        except Exception as e:
            logger.warning(f"All extractors failed: {e}")
        
        # Try with just simple extractor
        simple_extractors = [ext for ext in extractors if isinstance(ext, SimpleLLMPathExtractor)]
        if simple_extractors:
            try:
                logger.info("Trying with just simple extractor...")
                index = PropertyGraphIndex.from_documents(
                    documents,
                    kg_extractors=simple_extractors[:1],
                    storage_context=self.storage_context,
                    embed_kg_nodes=True,
                    show_progress=True,
                )
                return index
            except Exception as e:
                logger.warning(f"Simple extractor failed: {e}")
        
        # Last resort: try without embedding
        try:
            logger.info("Last resort: trying without node embedding...")
            index = PropertyGraphIndex.from_documents(
                documents,
                kg_extractors=extractors[:1],
                storage_context=self.storage_context,
                embed_kg_nodes=False,
                show_progress=True,
            )
            return index
        except Exception as e:
            logger.error(f"All fallback strategies failed: {e}")
            raise
    
    def _add_basic_enhancements(self, documents: List[Document]):
        """Add basic enhancements with error handling"""
        try:
            logger.info("Adding basic graph enhancements...")
            driver = get_neo4j_driver()
            
            with driver.session() as session:
                # Create basic document nodes
                self._create_basic_document_nodes(session, documents)
                
                # Create basic fiscal year nodes
                self._create_basic_fiscal_year_nodes(session, documents)
                
                # Add basic constraints
                self._add_basic_constraints(session)
                
            logger.info("✅ Basic enhancements completed")
            
        except Exception as e:
            logger.warning(f"Graph enhancements failed: {e}")
            # Don't fail the whole process for enhancement failures
    
    def _create_basic_document_nodes(self, session, documents: List[Document]):
        """Create basic document nodes"""
        processed_files = set()
        
        for doc in documents:
            file_name = doc.metadata.get('file_name', 'Unknown')
            
            if file_name not in processed_files and file_name != 'Unknown':
                try:
                    query = """
                    MERGE (d:BUDGET_DOCUMENT {name: $file_name})
                    SET d.processed_date = datetime()
                    """
                    session.run(query, file_name=file_name)
                    processed_files.add(file_name)
                except Exception as e:
                    logger.debug(f"Failed to create document node for {file_name}: {e}")
        
        logger.info(f"Created {len(processed_files)} document nodes")
    
    def _create_basic_fiscal_year_nodes(self, session, documents: List[Document]):
        """Create basic fiscal year nodes"""
        years = set()
        
        for doc in documents:
            doc_year = doc.metadata.get('document_year')
            if isinstance(doc_year, int) and 2020 <= doc_year <= 2030:
                years.add(doc_year)
        
        for year in years:
            try:
                query = """
                MERGE (fy:FISCAL_YEAR {year: $year})
                SET fy.name = $year_name
                """
                session.run(query, year=year, year_name=f"FY {year}")
            except Exception as e:
                logger.debug(f"Failed to create year node for {year}: {e}")
        
        logger.info(f"Created {len(years)} fiscal year nodes")
    
    def _add_basic_constraints(self, session):
        """Add basic constraints with error handling"""
        constraints = [
            "CREATE CONSTRAINT budget_doc_unique IF NOT EXISTS FOR (d:BUDGET_DOCUMENT) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT fiscal_year_unique IF NOT EXISTS FOR (fy:FISCAL_YEAR) REQUIRE fy.year IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                logger.debug(f"Constraint creation failed (may already exist): {e}")

class SimpleGraphBuilder:
    """Simple, reliable graph builder as fallback"""
    
    def __init__(self):
        self.graph_store = get_graph_store()
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
    
    def build_simple_graph(self, documents: List[Document]) -> PropertyGraphIndex:
        """Build simple graph with minimal extractors"""
        logger.info("Building simple knowledge graph...")
        
        # Severely limit documents and clean them
        limited_docs = documents[:20]  # Only 20 documents
        cleaned_docs = []
        
        for doc in limited_docs:
            cleaned_doc = Document(
                text=doc.get_content()[:1500],  # Limit content
                metadata={
                    'file_name': doc.metadata.get('file_name', 'unknown')[:20],
                    'source': 'budget_doc'
                }
            )
            cleaned_docs.append(cleaned_doc)
        
        # Use only simple extractor
        try:
            extractor = SimpleLLMPathExtractor(llm=Settings.llm)
            
            index = PropertyGraphIndex.from_documents(
                cleaned_docs,
                kg_extractors=[extractor],
                storage_context=self.storage_context,
                embed_kg_nodes=False,  # Disable embedding for simplicity
                show_progress=True,
            )
            
            logger.info("✅ Simple knowledge graph created")
            return index
            
        except Exception as e:
            logger.error(f"Simple graph creation failed: {e}")
            raise

def build_graph_from_documents(documents: List[Document]) -> PropertyGraphIndex:
    """Main function with progressive fallback strategy"""
    if not documents:
        logger.error("No documents provided for graph construction")
        return None
    
    logger.info(f"Building graph from {len(documents)} documents")
    
    # Try enhanced builder first
    try:
        builder = EnhancedGraphBuilder()
        return builder.build_graph_with_multiple_extractors(documents)
        
    except Exception as e:
        logger.warning(f"Enhanced graph construction failed: {e}")
        
        # Fallback to simple builder
        try:
            logger.info("Falling back to simple graph builder...")
            simple_builder = SimpleGraphBuilder()
            return simple_builder.build_simple_graph(documents)
            
        except Exception as simple_error:
            logger.error(f"Simple graph construction also failed: {simple_error}")
            
            # Return None rather than crashing - graph is optional
            logger.info("Graph construction completely failed - continuing without graph")
            return None