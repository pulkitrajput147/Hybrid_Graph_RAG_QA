# 1_ingestion/document_parser.py

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.node_parser.text import TokenTextSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
    BaseExtractor
)
from llama_index.core.schema import BaseNode, TextNode
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import DATA_DIR, Settings

class BudgetDocumentExtractor(BaseExtractor):
    """Custom extractor for budget-specific metadata with size controls"""
    
    def extract(self, nodes: List) -> List[Dict[str, Any]]:
        metadata_list = []
        
        for node in nodes:
            try:
                content = node.get_content()
                metadata = {
                    "budget_year": self._extract_year(content),
                    "department": self._extract_department(content),
                    "budget_amount": self._extract_budget_amounts(content),
                    "document_type": self._classify_document_type(content),
                    "contains_table": self._has_table_markers(content),
                    "contains_financial_data": self._has_financial_data(content)
                }
                
                # Apply strict size limits to prevent issues
                metadata = self._limit_metadata_size(metadata)
                metadata_list.append(metadata)
                
            except Exception as e:
                logger.warning(f"Metadata extraction failed for node: {e}")
                metadata_list.append({})  # Empty metadata for failed extractions
        
        return metadata_list
    
    def _limit_metadata_size(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strict metadata size limits"""
        limited_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, list):
                # Limit to max 3 items, each max 30 chars
                if len(value) > 3:
                    value = value[:3]
                
                limited_list = []
                for item in value:
                    if isinstance(item, str):
                        if len(item) > 30:
                            limited_list.append(item[:30])
                        else:
                            limited_list.append(item)
                    else:
                        limited_list.append(item)
                limited_metadata[key] = limited_list
                
            elif isinstance(value, str):
                # Limit strings to 50 chars max
                if len(value) > 50:
                    limited_metadata[key] = value[:50]
                else:
                    limited_metadata[key] = value
            else:
                limited_metadata[key] = value
        
        return limited_metadata
    
    async def aextract(self, nodes: List) -> List[Dict[str, Any]]:
        """Async version - fallback to sync for simplicity"""
        return self.extract(nodes)
    
    def _extract_year(self, content: str) -> List[int]:
        """Extract budget years from content"""
        try:
            years = []
            year_pattern = r'\b(20[2-3][0-9])\b'
            matches = re.findall(year_pattern, content[:1000])  # Only check first 1000 chars
            unique_years = list(set(int(year) for year in matches))
            return unique_years[:2]  # Max 2 years
        except:
            return []
    
    def _extract_department(self, content: str) -> List[str]:
        """Extract department names with strict limits"""
        try:
            departments = []
            dept_patterns = [
                r'Department of (\w+(?:\s+\w+)?)',  # Max 2 words
                r'Ministry of (\w+(?:\s+\w+)?)',    # Max 2 words
                r'(\w+) Department',
                r'(\w+) Ministry'
            ]
            
            # Only check first 2000 chars for performance
            content_sample = content[:2000]
            
            for pattern in dept_patterns:
                matches = re.findall(pattern, content_sample, re.IGNORECASE)
                departments.extend(matches[:2])  # Max 2 per pattern
                
            # Clean and limit
            unique_depts = list(set(departments))[:2]  # Max 2 departments
            return [dept.strip()[:25] for dept in unique_depts if dept.strip()]
        except:
            return []
    
    def _extract_budget_amounts(self, content: str) -> List[str]:
        """Extract monetary amounts with strict limits"""
        try:
            money_patterns = [
                r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
                r'[\d,]+(?:\.\d{2})?\s*(?:million|billion|trillion)?\s*dollars?'
            ]
            
            amounts = []
            content_sample = content[:1000]  # Only check first 1000 chars
            
            for pattern in money_patterns:
                matches = re.findall(pattern, content_sample, re.IGNORECASE)
                amounts.extend(matches[:2])  # Max 2 per pattern
            
            unique_amounts = list(set(amounts))[:3]  # Max 3 amounts
            return [amt.strip()[:15] for amt in unique_amounts if amt.strip()]
        except:
            return []
    
    def _classify_document_type(self, content: str) -> str:
        """Classify document type"""
        try:
            content_lower = content[:500].lower()  # Only check first 500 chars
            
            if any(term in content_lower for term in ['executive summary', 'overview']):
                return 'summary'
            elif any(term in content_lower for term in ['detailed budget', 'line item']):
                return 'detailed'
            elif any(term in content_lower for term in ['analysis', 'comparison']):
                return 'analysis'
            elif any(term in content_lower for term in ['appendix', 'supplement']):
                return 'appendix'
            else:
                return 'general'
        except:
            return 'unknown'
    
    def _has_table_markers(self, content: str) -> bool:
        """Check for table markers"""
        try:
            content_sample = content[:1000]
            table_indicators = [
                r'\|\s*\w+\s*\|',
                r'\t.*\t.*\t',
                r'Total\s*\$'
            ]
            return any(re.search(pattern, content_sample) for pattern in table_indicators)
        except:
            return False
    
    def _has_financial_data(self, content: str) -> bool:
        """Check for financial data"""
        try:
            financial_terms = ['budget', 'allocation', 'expenditure', 'revenue', 'fiscal']
            content_lower = content[:500].lower()
            return any(term in content_lower for term in financial_terms)
        except:
            return False

class MultiModalDocumentParser:
    """Enhanced document parser with better error handling"""
    
    def __init__(self):
        # Initialize with fewer extractors to reduce complexity
        self.extractors = [
            BudgetDocumentExtractor()
        ]
    
    def parse_with_multiple_strategies(self, file_path: Path) -> List[Document]:
        """Try multiple parsing strategies with better error handling"""
        strategies = [
            ("pymupdf", self._parse_with_pymupdf),
            ("simple", self._parse_with_simple),
            ("unstructured", self._parse_with_unstructured),
            ("docling", self._parse_with_docling)
        ]
        
        best_docs = []
        best_score = 0
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Trying {strategy_name} parser for {file_path.name}")
                docs = strategy_func(file_path)
                
                if docs:
                    score = self._evaluate_parsing_quality(docs)
                    logger.info(f"{strategy_name} parser score: {score}")
                    
                    if score > best_score:
                        best_docs = docs
                        best_score = score
                        logger.info(f"New best parser: {strategy_name}")
                        
                    # If we get a good enough score, use it
                    if score > 10:
                        logger.info(f"Using {strategy_name} parser (good enough score)")
                        break
                        
            except Exception as e:
                logger.warning(f"{strategy_name} parser failed for {file_path.name}: {e}")
                continue
        
        if not best_docs:
            logger.error(f"All parsing strategies failed for {file_path.name}")
            return []
            
        return self._enhance_documents(best_docs, file_path)
    
    def _parse_with_docling(self, file_path: Path) -> List[Document]:
        """Parse using Docling reader"""
        try:
            reader = DoclingReader()
            docs = reader.load_data(file_path=str(file_path))
            return docs
        except Exception as e:
            logger.debug(f"Docling parsing failed: {e}")
            return []
    
    def _parse_with_unstructured(self, file_path: Path) -> List[Document]:
        """Parse using Unstructured reader"""
        try:
            reader = UnstructuredReader()
            docs = reader.load_data(file_path=str(file_path))
            return docs
        except Exception as e:
            logger.debug(f"Unstructured parsing failed: {e}")
            return []
    
    def _parse_with_pymupdf(self, file_path: Path) -> List[Document]:
        """Parse using PyMuPDF - most reliable"""
        try:
            reader = PyMuPDFReader()
            docs = reader.load_data(file_path=str(file_path))
            return docs
        except Exception as e:
            logger.debug(f"PyMuPDF parsing failed: {e}")
            return []
    
    def _parse_with_simple(self, file_path: Path) -> List[Document]:
        """Fallback simple parsing"""
        try:
            reader = SimpleDirectoryReader(input_files=[str(file_path)])
            docs = reader.load_data()
            return docs
        except Exception as e:
            logger.debug(f"Simple parsing failed: {e}")
            return []
    
    def _evaluate_parsing_quality(self, docs: List[Document]) -> float:
        """Evaluate parsing quality"""
        if not docs:
            return 0
        
        total_score = 0
        
        for doc in docs:
            try:
                content = doc.get_content()
                score = 0
                
                # Length score
                length_score = min(len(content) / 1000, 5)
                score += length_score
                
                # Structure indicators
                if any(marker in content for marker in ['Table', 'Figure', 'Chart']):
                    score += 3
                
                # Financial content
                if re.search(r'\$[\d,]+', content):
                    score += 3
                
                # Clean text (not too many special chars)
                if content and len(content) > 0:
                    special_ratio = len(re.findall(r'[^\w\s\.\,\!\?]', content)) / len(content)
                    if special_ratio < 0.2:
                        score += 2
                
                total_score += score
            except Exception as e:
                logger.debug(f"Error evaluating doc quality: {e}")
                continue
        
        return total_score / len(docs) if docs else 0
    
    def _enhance_documents(self, docs: List[Document], file_path: Path) -> List[Document]:
        """Enhance documents with basic metadata"""
        enhanced_docs = []
        
        for i, doc in enumerate(docs):
            try:
                # Add minimal essential metadata
                doc.metadata.update({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'chunk_id': i,
                    'total_chunks': len(docs),
                })
                
                # Extract year from filename
                year_match = re.search(r'(20[2-3][0-9])', file_path.name)
                if year_match:
                    doc.metadata['document_year'] = int(year_match.group(1))
                
                enhanced_docs.append(doc)
            except Exception as e:
                logger.warning(f"Error enhancing document {i}: {e}")
                enhanced_docs.append(doc)  # Add without enhancement
        
        return enhanced_docs

def load_and_parse_documents(data_path: str = DATA_DIR) -> List[Document]:
    """Load and parse documents with improved error handling"""
    logger.info(f"Starting document parsing from: {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data directory '{data_path}' not found.")
        return []

    parser = MultiModalDocumentParser()
    all_documents = []
    
    # Find all PDF files
    pdf_files = list(Path(data_path).glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_path}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        
        try:
            docs = parser.parse_with_multiple_strategies(pdf_file)
            if docs:
                # Split documents with safe chunk sizes
                docs = split_documents_intelligently(docs)
                # Apply limited metadata extraction
                docs = apply_metadata_extraction(docs, parser.extractors)
                all_documents.extend(docs)
                logger.info(f"Successfully processed {pdf_file.name}: {len(docs)} chunks")
            else:
                logger.warning(f"No content extracted from {pdf_file.name}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            continue
    
    logger.info(f"Total documents processed: {len(all_documents)}")
    return all_documents

def split_documents_intelligently(docs: List[Document]) -> List[Document]:
    """Split documents with proper chunk size management"""
    try:
        # Calculate safe chunk size based on document content
        max_doc_size = max((len(doc.get_content()) for doc in docs), default=0)
        
        # Use adaptive chunk size
        if max_doc_size > 10000:
            chunk_size = 1500
        elif max_doc_size > 5000:
            chunk_size = 1000  
        else:
            chunk_size = 800
            
        logger.info(f"Using chunk size: {chunk_size} for documents with max size: {max_doc_size}")
        
        # Use sentence splitter as primary method
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(150, chunk_size // 6),
        )
        
        nodes = splitter.get_nodes_from_documents(docs)
        
        # Convert back to documents
        split_docs = []
        for node in nodes:
            doc = Document(
                text=node.get_content(),
                metadata=node.metadata.copy() if hasattr(node, 'metadata') else {}
            )
            split_docs.append(doc)
            
        return split_docs
        
    except Exception as e:
        logger.warning(f"Intelligent splitting failed: {e}. Using fallback.")
        
        # Fallback to simple token splitting
        splitter = TokenTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        nodes = splitter.get_nodes_from_documents(docs)
        
        split_docs = []
        for node in nodes:
            doc = Document(
                text=node.get_content(),
                metadata=getattr(node, 'metadata', {})
            )
            split_docs.append(doc)
            
        return split_docs

def apply_metadata_extraction(docs: List[Document], extractors: List[BaseExtractor]) -> List[Document]:
    """Apply metadata extraction with proper error handling"""
    if not extractors:
        logger.info("No extractors provided, skipping metadata extraction")
        return docs
        
    try:
        # Convert documents to nodes
        nodes = []
        for doc in docs:
            try:
                text_content = doc.get_content()
                
                # Skip empty documents
                if not text_content.strip():
                    continue
                    
                node = TextNode(
                    text=text_content,
                    metadata=doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                )
                nodes.append(node)
            except Exception as e:
                logger.warning(f"Failed to convert document to node: {e}")
                continue
        
        if not nodes:
            logger.warning("No valid nodes created from documents")
            return docs
        
        # Apply extractors one by one with error handling
        for extractor in extractors:
            try:
                logger.info(f"Applying {extractor.__class__.__name__}")
                metadata_list = extractor.extract(nodes)
                
                # Update node metadata
                for node, metadata in zip(nodes, metadata_list):
                    if metadata:
                        if not hasattr(node, 'metadata'):
                            node.metadata = {}
                        
                        # Merge new metadata carefully
                        for key, value in metadata.items():
                            if value:  # Only add non-empty values
                                node.metadata[key] = value
                
                logger.info(f"✅ {extractor.__class__.__name__} completed")
                        
            except Exception as e:
                logger.warning(f"Extractor {extractor.__class__.__name__} failed: {e}")
                continue
        
        # Convert back to documents
        enhanced_docs = []
        for node in nodes:
            try:
                doc = Document(
                    text=node.get_content(),
                    metadata=getattr(node, 'metadata', {})
                )
                enhanced_docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to convert node back to document: {e}")
                continue
        
        return enhanced_docs if enhanced_docs else docs
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return docs