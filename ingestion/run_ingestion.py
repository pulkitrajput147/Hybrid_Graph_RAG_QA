# 1_ingestion/run_ingestion.py

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import argparse
import logging
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from document_parser import load_and_parse_documents
from graph_builder import build_graph_from_documents
from vectorizer import build_vector_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class RobustIngestionPipeline:
    """
    Robust ingestion pipeline with comprehensive error handling and recovery
    """
    
    def __init__(self, data_path: str, vector_strategy: str = "enhanced"):
        self.data_path = data_path
        self.vector_strategy = vector_strategy
        self.documents = []
        self.graph_index = None
        self.vector_index = None
        self.stats = {
            'start_time': None,
            'end_time': None,
            'documents_processed': 0,
            'total_chunks': 0,
            'parsing_errors': [],
            'graph_errors': [],
            'vector_errors': [],
            'success': False
        }
    
    async def run_complete_pipeline(self):
        """Run the complete pipeline with comprehensive error handling"""
        self.stats['start_time'] = datetime.now()
        logger.info("🚀 Starting Robust Budget Document Ingestion Pipeline")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Vector strategy: {self.vector_strategy}")
        
        pipeline_success = True
        
        try:
            # Stage 1: Document Parsing (Critical - must succeed)
            parsing_success = await self._run_document_parsing_safe()
            if not parsing_success:
                logger.error("❌ Document parsing failed - cannot continue")
                return False
            
            # Stage 2: Vector Index Construction (Critical - must succeed)
            vector_success = await self._run_vector_construction_safe()
            if not vector_success:
                logger.error("❌ Vector construction failed - pipeline failed")
                pipeline_success = False
            
            # Stage 3: Knowledge Graph Construction (Optional - can fail)
            graph_success = await self._run_graph_construction_safe()
            if not graph_success:
                logger.warning("⚠️ Graph construction failed - continuing without graph")
            
            # Stage 4: Validation (Optional)
            await self._run_validation_safe()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with unexpected error: {e}")
            logger.error(traceback.format_exc())
            pipeline_success = False
        finally:
            self.stats['success'] = pipeline_success
            self._finalize_pipeline()
        
        return pipeline_success
    
    async def _run_document_parsing_safe(self) -> bool:
        """Stage 1: Parse documents with comprehensive error handling"""
        logger.info("\n" + "="*60)
        logger.info("📄 STAGE 1: Document Parsing (Critical)")
        logger.info("="*60)
        
        try:
            logger.info("Attempting document parsing...")
            self.documents = load_and_parse_documents(self.data_path)
            
            if not self.documents:
                logger.error("No documents were successfully parsed")
                self.stats['parsing_errors'].append("No documents parsed successfully")
                return False
            
            # Calculate stats
            file_names = set(doc.metadata.get('file_name', 'unknown') for doc in self.documents)
            self.stats['documents_processed'] = len(file_names)
            self.stats['total_chunks'] = len(self.documents)
            
            logger.info(f"✅ Successfully parsed {self.stats['documents_processed']} files into {self.stats['total_chunks']} chunks")
            
            # Validate parsed documents
            valid_docs = []
            for doc in self.documents:
                try:
                    content = doc.get_content()
                    if content and len(content.strip()) > 10:
                        valid_docs.append(doc)
                except Exception as e:
                    logger.debug(f"Skipping invalid document: {e}")
            
            self.documents = valid_docs
            logger.info(f"Validated {len(self.documents)} documents")
            
            if len(self.documents) == 0:
                logger.error("No valid documents after validation")
                return False
            
            self._log_document_stats()
            return True
            
        except Exception as e:
            error_msg = f"Document parsing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            self.stats['parsing_errors'].append(error_msg)
            return False
    
    async def _run_vector_construction_safe(self) -> bool:
        """Stage 2: Build vector index (Critical)"""
        logger.info("\n" + "="*60) 
        logger.info("🔍 STAGE 2: Vector Index Construction (Critical)")
        logger.info("="*60)
        
        try:
            logger.info(f"Building vector index with strategy: {self.vector_strategy}")
            self.vector_index = build_vector_index(self.documents, strategy=self.vector_strategy)
            
            if self.vector_index:
                logger.info("✅ Vector index construction completed successfully")
                await self._validate_vector_index_safe()
                return True
            else:
                error_msg = "Vector index construction returned None"
                logger.error(f"❌ {error_msg}")
                self.stats['vector_errors'].append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Vector construction failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            self.stats['vector_errors'].append(error_msg)
            
            # Try with simple strategy as last resort
            if self.vector_strategy != "simple":
                logger.info("🔄 Attempting fallback to simple strategy...")
                try:
                    self.vector_index = build_vector_index(self.documents, strategy="simple")
                    if self.vector_index:
                        logger.info("✅ Simple vector index created as fallback")
                        return True
                except Exception as fallback_error:
                    fallback_msg = f"Fallback vector construction also failed: {str(fallback_error)}"
                    logger.error(f"❌ {fallback_msg}")
                    self.stats['vector_errors'].append(fallback_msg)
            
            return False
    
    async def _run_graph_construction_safe(self) -> bool:
        """Stage 3: Build knowledge graph (Optional)"""
        logger.info("\n" + "="*60)
        logger.info("🕸️  STAGE 3: Knowledge Graph Construction (Optional)")
        logger.info("="*60)
        
        try:
            logger.info("Building knowledge graph...")
            self.graph_index = build_graph_from_documents(self.documents)
            
            if self.graph_index:
                logger.info("✅ Knowledge graph construction completed successfully")
                await self._validate_graph_safe()
                return True
            else:
                logger.warning("⚠️ Graph construction returned None")
                self.stats['graph_errors'].append("Graph construction returned None")
                return False
                
        except Exception as e:
            error_msg = f"Graph construction failed: {str(e)}"
            logger.warning(f"⚠️ {error_msg}")
            logger.debug(traceback.format_exc())  # Debug level for optional component
            self.stats['graph_errors'].append(error_msg)
            return False
    
    async def _run_validation_safe(self):
        """Stage 4: Validate the ingested data (Optional)"""
        logger.info("\n" + "="*60)
        logger.info("✅ STAGE 4: Data Validation (Optional)")
        logger.info("="*60)
        
        try:
            # Test vector store
            if self.vector_index:
                logger.info("   ✅ Vector store: Available")
            else:
                logger.warning("   ⚠️ Vector store: Not available")
            
            # Test graph store  
            if self.graph_index:
                logger.info("   ✅ Graph store: Available")
            else:
                logger.warning("   ⚠️ Graph store: Not available")
            
            # Basic data integrity
            if self.documents:
                empty_docs = sum(1 for doc in self.documents if not doc.get_content().strip())
                if empty_docs > 0:
                    logger.warning(f"   ⚠️ Found {empty_docs} empty document chunks")
                else:
                    logger.info("   ✅ All documents have content")
            
            logger.info("✅ Validation completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Validation failed: {e}")
    
    async def _validate_vector_index_safe(self):
        """Safely validate vector index"""
        try:
            if hasattr(self.vector_index, 'docstore') and self.vector_index.docstore:
                node_count = len(self.vector_index.docstore.docs)
                logger.info(f"   Vector index nodes: {node_count}")
            else:
                logger.info("   Vector index: Created (node count unavailable)")
        except Exception as e:
            logger.debug(f"Vector validation details unavailable: {e}")
    
    async def _validate_graph_safe(self):
        """Safely validate graph"""
        try:
            from core.config import get_neo4j_driver
            
            driver = get_neo4j_driver()
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()['node_count']
                logger.info(f"   Graph nodes: {node_count}")
                
        except Exception as e:
            logger.debug(f"Graph validation failed: {e}")
    
    def _log_document_stats(self):
        """Log document statistics safely"""
        try:
            file_stats = {}
            year_stats = {}
            
            for doc in self.documents:
                file_name = doc.metadata.get('file_name', 'unknown')
                doc_year = doc.metadata.get('document_year', 'unknown')
                
                file_stats[file_name] = file_stats.get(file_name, 0) + 1
                year_stats[doc_year] = year_stats.get(doc_year, 0) + 1
            
            logger.info("📊 Document Statistics:")
            logger.info(f"   Files processed: {len(file_stats)}")
            
            # Get valid years only
            valid_years = [y for y in year_stats.keys() if isinstance(y, int) and 2000 <= y <= 2030]
            if valid_years:
                logger.info(f"   Years covered: {sorted(valid_years)}")
            else:
                logger.info("   Years covered: Unknown")
            
            # Show sample file stats
            sample_files = dict(list(file_stats.items())[:3])
            logger.info(f"   Sample chunks per file: {sample_files}")
            
        except Exception as e:
            logger.debug(f"Error logging document stats: {e}")
    
    def _finalize_pipeline(self):
        """Finalize pipeline and log comprehensive summary"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("\n" + "="*60)
        logger.info("🎉 PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Files processed: {self.stats['documents_processed']}")
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        
        # Error summary
        total_errors = len(self.stats['parsing_errors']) + len(self.stats['vector_errors']) + len(self.stats['graph_errors'])
        logger.info(f"Total errors: {total_errors}")
        
        # Component status
        logger.info("\n📊 Component Status:")
        logger.info(f"   Document Parsing: {'✅ SUCCESS' if len(self.stats['parsing_errors']) == 0 else '❌ FAILED'}")
        logger.info(f"   Vector Index: {'✅ SUCCESS' if len(self.stats['vector_errors']) == 0 else '❌ FAILED'}")
        logger.info(f"   Knowledge Graph: {'✅ SUCCESS' if len(self.stats['graph_errors']) == 0 else '⚠️ FAILED (Optional)'}")
        
        # Error details
        if total_errors > 0:
            logger.info("\n⚠️ Error Details:")
            for error in self.stats['parsing_errors']:
                logger.warning(f"   Parsing: {error}")
            for error in self.stats['vector_errors']:
                logger.error(f"   Vector: {error}")
            for error in self.stats['graph_errors']:
                logger.warning(f"   Graph: {error}")
        
        # Final status
        if self.stats['success']:
            logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("Ready for querying with available components.")
        else:
            logger.error("\n❌ PIPELINE COMPLETED WITH CRITICAL FAILURES")
            logger.error("Vector index is required for basic functionality.")

async def main():
    """Main function with enhanced CLI support"""
    parser = argparse.ArgumentParser(description="Robust Budget Document Ingestion Pipeline")
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--vector-strategy", choices=["simple", "enhanced"], 
                       default="enhanced", help="Vector indexing strategy")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Determine data path
    data_path = args.data_path
    if not data_path:
        try:
            from core.config import DATA_DIR
            data_path = DATA_DIR
        except ImportError:
            logger.error("Could not import DATA_DIR and no data-path provided")
            return 1
    
    # Validate data path
    if not os.path.exists(data_path):
        logger.error(f"Data directory does not exist: {data_path}")
        return 1
    
    # Check for PDF files
    pdf_files = list(Path(data_path).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {data_path}")
        return 1
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Create and run pipeline
    pipeline = RobustIngestionPipeline(data_path, args.vector_strategy)
    
    try:
        success = await pipeline.run_complete_pipeline()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    # Handle asyncio event loop issues
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    exit_code = asyncio.run(main())
    sys.exit(exit_code)