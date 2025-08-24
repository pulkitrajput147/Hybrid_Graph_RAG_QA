# core/config.py

import os
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from typing import Dict, Any


import qdrant_client
import neo4j

# Load environment variables from .env file
load_dotenv()

@lru_cache(maxsize=1)
def get_neo4j_driver():
    """Returns a singleton Neo4j driver instance."""
    return neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

@lru_cache(maxsize=1)
def get_qdrant_client():
    """Returns a singleton Qdrant client instance."""
    return qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    )

def configure_global_settings():
    """Configures and applies global LlamaIndex settings."""
    llm = OpenAI(model="gpt-4.1", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20

# Call this at application startup
configure_global_settings()

# Graph Store Configuration
def get_graph_store() -> Neo4jPropertyGraphStore:
    return Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database="neo4j",
    )
# Vector Store Configuration
def get_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME")
    )

# Constants
DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")


# Enhanced configuration for dynamic ingestion
class IngestionConfig:
    """Configuration class for dynamic ingestion pipeline"""
    
    # Document parsing configurations
    PARSING_STRATEGIES = {
        "docling": {
            "priority": 1,
            "best_for": ["complex_layouts", "tables", "figures"],
            "timeout": 300
        },
        "unstructured": {
            "priority": 2, 
            "best_for": ["mixed_content", "scanned_docs"],
            "timeout": 180,
            "strategy": "hi_res"
        },
        "pymupdf": {
            "priority": 3,
            "best_for": ["text_heavy", "simple_layouts"],
            "timeout": 60
        },
        "simple": {
            "priority": 4,
            "best_for": ["fallback"],
            "timeout": 30
        }
    }
    
    # Text splitting configurations
    CHUNKING_CONFIG = {
        "semantic": {
            "buffer_size": 1,
            "breakpoint_percentile_threshold": 95,
            "min_chunk_size": 200
        },
        "sentence": {
            "chunk_size": 1024,
            "chunk_overlap": 200,
            "paragraph_separator": "\n\n\n"
        },
        "token": {
            "chunk_size": 1024,
            "chunk_overlap": 200
        }
    }
    
    # Graph extraction configurations
    GRAPH_CONFIG = {
        "schema_extractor": {
            "max_triplets_per_chunk": 5,
            "num_workers": 4,
            "strict": False
        },
        "dynamic_extractor": {
            "max_triplets_per_chunk": 5,
            "num_workers": 4
        },
        "simple_extractor": {
            "max_triplets_per_chunk": 3,
            "num_workers": 2
        }
    }
    
    # Vector indexing configurations  
    VECTOR_CONFIG = {
        "enhanced": {
            "use_ingestion_pipeline": True,
            "extractors": ["title", "keywords", "summary", "questions"],
            "chunk_strategy": "semantic"
        },
        "simple": {
            "use_ingestion_pipeline": False,
            "chunk_strategy": "sentence"
        },
        "hybrid": {
            "create_specialized_indices": True,
            "representations": ["full", "financial", "departmental"]
        }
    }
    
    # File processing configurations
    FILE_CONFIG = {
        "supported_formats": [".pdf"],
        "max_file_size_mb": 100,
        "skip_empty_files": True,
        "backup_failed_files": True
    }

# Environment-based configuration override
def get_ingestion_config() -> Dict[str, Any]:
    """Get ingestion configuration with environment overrides"""
    config = {
        "parsing_timeout": int(os.getenv("PARSING_TIMEOUT", "300")),
        "max_workers": int(os.getenv("MAX_WORKERS", "4")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1024")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "enable_ocr": os.getenv("ENABLE_OCR", "false").lower() == "true",
        "validate_after_ingestion": os.getenv("VALIDATE_INGESTION", "true").lower() == "true"
    }
    return config