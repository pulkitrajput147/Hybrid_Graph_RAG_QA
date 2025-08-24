# 1_ingestion/vectorizer.py

from typing import List, Dict, Any
import logging
from datetime import datetime

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser.text import TokenTextSplitter
from llama_index.core.schema import MetadataMode
import numpy as np

from core.config import get_vector_store, Settings

logger = logging.getLogger(__name__)

class BudgetVectorProcessor:
    """Enhanced vector processor with proper metadata handling"""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
    def create_enhanced_vector_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create vector index with safe processing pipeline"""
        logger.info(f"Creating enhanced vector index from {len(documents)} documents")
        
        # Clean and validate documents
        cleaned_documents = self._clean_and_validate_documents(documents)
        
        if not cleaned_documents:
            raise ValueError("No valid documents after cleaning")
        
        # Process documents through safe pipeline
        processed_nodes = self._create_safe_processing_pipeline(cleaned_documents)
        
        logger.info(f"Processed into {len(processed_nodes)} nodes")
        
        # Create vector index from processed nodes
        try:
            index = VectorStoreIndex(
                processed_nodes,
                storage_context=self.storage_context,
                embed_model=Settings.embed_model,
                show_progress=True
            )
            
            logger.info("✅ Enhanced vector index creation completed")
            return index
            
        except Exception as e:
            logger.error(f"Vector index creation failed: {e}")
            # Try with simplified approach
            return self._create_simple_fallback_index(cleaned_documents)
    
    def _clean_and_validate_documents(self, documents: List[Document]) -> List[Document]:
        """Clean and validate documents before processing"""
        logger.info("Cleaning and validating documents...")
        
        cleaned_documents = []
        
        for i, doc in enumerate(documents):
            try:
                # Get content and validate
                content = doc.get_content()
                if not content or len(content.strip()) < 10:
                    logger.debug(f"Skipping document {i}: too short or empty")
                    continue
                
                # Clean metadata - keep only essential fields
                cleaned_metadata = self._clean_metadata(doc.metadata)
                
                # Create cleaned document
                cleaned_doc = Document(
                    text=content,
                    metadata=cleaned_metadata
                )
                
                cleaned_documents.append(cleaned_doc)
                
            except Exception as e:
                logger.warning(f"Failed to clean document {i}: {e}")
                continue
        
        logger.info(f"Cleaned {len(cleaned_documents)} documents from {len(documents)} originals")
        return cleaned_documents
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to essential fields only"""
        essential_fields = [
            'file_name', 'document_year', 'document_type', 'chunk_id', 
            'contains_financial_data', 'budget_year', 'department'
        ]
        
        cleaned_metadata = {}
        
        for key, value in metadata.items():
            if key in essential_fields:
                if isinstance(value, str):
                    # Truncate strings
                    cleaned_metadata[key] = value[:50] if len(value) > 50 else value
                elif isinstance(value, list):
                    # Limit lists
                    if len(value) > 3:
                        value = value[:3]
                    cleaned_metadata[key] = value
                elif isinstance(value, (int, float, bool)):
                    cleaned_metadata[key] = value
                # Skip other types
        
        # Add processing metadata
        cleaned_metadata['processed_at'] = datetime.now().strftime("%Y-%m-%d")
        
        return cleaned_metadata
    
    def _create_safe_processing_pipeline(self, documents: List[Document]) -> List:
        """Create a safe processing pipeline without LLM extractors"""
        logger.info("Setting up safe document processing pipeline...")
        
        # Calculate safe chunk size
        avg_content_length = np.mean([len(doc.get_content()) for doc in documents])
        max_metadata_size = max(len(str(doc.metadata)) for doc in documents) if documents else 0
        
        # Conservative chunk size calculation
        safe_chunk_size = min(1500, max(800, int(avg_content_length / 3)))
        safe_chunk_size = max(safe_chunk_size, max_metadata_size + 300)
        
        logger.info(f"Using chunk size: {safe_chunk_size} (avg content: {avg_content_length:.0f}, max metadata: {max_metadata_size})")
        
        # Use simple splitting without LLM extractors to avoid metadata issues
        transformations = [
            SentenceSplitter(
                chunk_size=safe_chunk_size,
                chunk_overlap=min(150, safe_chunk_size // 10),
                paragraph_separator="\n\n\n",
                secondary_chunking_regex="[.!?]\\s+",
            )
        ]
        
        # Create pipeline
        try:
            pipeline = IngestionPipeline(
                transformations=transformations,
                vector_store=self.vector_store
            )
            
            processed_nodes = pipeline.run(documents=documents, show_progress=True)
            return processed_nodes
            
        except Exception as e:
            logger.warning(f"Pipeline processing failed: {e}. Using fallback.")
            return self._simple_processing_fallback(documents, safe_chunk_size)
    
    def _simple_processing_fallback(self, documents: List[Document], chunk_size: int = 1200) -> List:
        """Simple fallback processing"""
        logger.info(f"Using simple processing fallback with chunk size {chunk_size}")
        
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(120, chunk_size // 10),
        )
        
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        
        # Add minimal processing metadata
        for i, node in enumerate(nodes):
            try:
                if not hasattr(node, 'metadata'):
                    node.metadata = {}
                node.metadata.update({
                    'node_id': f"node_{i}",
                    'processing_method': 'fallback',
                })
            except Exception as e:
                logger.debug(f"Failed to add metadata to node {i}: {e}")
        
        return nodes
    
    def _create_simple_fallback_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create simple fallback index"""
        logger.info("Creating simple fallback vector index...")
        
        # Use very conservative chunk size
        safe_chunk_size = 1000
        
        splitter = TokenTextSplitter(
            chunk_size=safe_chunk_size,
            chunk_overlap=100
        )
        
        nodes = splitter.get_nodes_from_documents(documents)
        
        index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            embed_model=Settings.embed_model,
            show_progress=True
        )
        
        logger.info("✅ Simple fallback vector index created")
        return index

class SimpleVectorProcessor:
    """Simplified vector processor for reliable operation"""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
    
    def create_simple_vector_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create simple, reliable vector index"""
        logger.info(f"Creating simple vector index from {len(documents)} documents")
        
        # Basic document validation
        valid_documents = []
        for doc in documents:
            try:
                content = doc.get_content()
                if content and len(content.strip()) > 10:
                    # Create document with minimal metadata
                    simple_doc = Document(
                        text=content,
                        metadata={
                            'file_name': doc.metadata.get('file_name', 'unknown'),
                            'document_year': doc.metadata.get('document_year', 'unknown'),
                            'chunk_id': doc.metadata.get('chunk_id', len(valid_documents))
                        }
                    )
                    valid_documents.append(simple_doc)
            except Exception as e:
                logger.debug(f"Skipping invalid document: {e}")
                continue
        
        if not valid_documents:
            raise ValueError("No valid documents found")
        
        # Simple chunking
        splitter = TokenTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
        )
        
        nodes = splitter.get_nodes_from_documents(valid_documents, show_progress=True)
        
        # Create index
        index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            embed_model=Settings.embed_model,
            show_progress=True
        )
        
        logger.info("✅ Simple vector index created successfully")
        return index

def build_vector_index(documents: List[Document], strategy: str = "enhanced") -> VectorStoreIndex:
    """Main function to build vector index with configurable strategies"""
    if not documents:
        logger.error("No documents provided for vector indexing")
        return None
    
    logger.info(f"Building vector index using '{strategy}' strategy")
    
    try:
        if strategy == "simple":
            processor = SimpleVectorProcessor()
            return processor.create_simple_vector_index(documents)
        elif strategy == "enhanced":
            processor = BudgetVectorProcessor()
            return processor.create_enhanced_vector_index(documents)
        else:
            # Default to enhanced
            processor = BudgetVectorProcessor()
            return processor.create_enhanced_vector_index(documents)
            
    except Exception as e:
        logger.error(f"Vector index creation failed with {strategy} strategy: {e}")
        
        # Always fallback to simple strategy
        logger.info("Falling back to simple strategy...")
        try:
            processor = SimpleVectorProcessor()
            return processor.create_simple_vector_index(documents)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise

def _build_simple_index(documents: List[Document]) -> VectorStoreIndex:
    """Simple vector index creation (legacy function for compatibility)"""
    processor = SimpleVectorProcessor()
    return processor.create_simple_vector_index(documents)