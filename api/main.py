# 3_api/main.py

import sys
from pathlib import Path
import logging
from typing import Optional, List
# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from functools import lru_cache
from enum import Enum
import uvicorn

from llama_index.core import Response
from retrieval.engine import create_query_engine, create_custom_decomposition_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid Graph RAG API",
    description="API for querying a multi-year budget analysis system with advanced query decomposition.",
    version="2.0.0",
)

class DecompositionType(str, Enum):
    """Available query decomposition methods."""
    none = "none"
    multistep = "multistep"
    subquestion = "subquestion"
    custom = "custom"

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query about budget documents")
    decomposition_type: DecompositionType = Field(
        default=DecompositionType.none,
        description="Type of query decomposition to use for complex queries"
    )
    max_retries: int = Field(default=3, ge=1, le=5, description="Maximum retry attempts")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    decomposition_type: str
    query_complexity: str
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    engines_loaded: dict

# Cache for different engine types
_engines_cache = {}

def get_query_engine(decomposition_type: DecompositionType = DecompositionType.none):
    """
    Get or create a cached query engine based on decomposition type.
    """
    if decomposition_type.value not in _engines_cache:
        logger.info(f"Loading {decomposition_type.value} query engine...")
        try:
            if decomposition_type == DecompositionType.custom:
                engine = create_custom_decomposition_engine()
            else:
                engine = create_query_engine(decomposition_type.value)
            _engines_cache[decomposition_type.value] = engine
            logger.info(f"✅ {decomposition_type.value} query engine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {decomposition_type.value} engine: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize {decomposition_type.value} query engine: {str(e)}"
            )
    
    return _engines_cache[decomposition_type.value]

def determine_query_complexity(query: str) -> str:
    """
    Simple heuristic to determine query complexity.
    """
    query_lower = query.lower()
    complexity_indicators = [
        'compare', 'analysis', 'percentage', 'change', 'between', 'and',
        'calculate', 'difference', 'trend', 'over time', 'year over year',
        'breakdown', 'detailed', 'comprehensive'
    ]
    
    # Count complexity indicators
    complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
    
    if complexity_score >= 3:
        return "high"
    elif complexity_score >= 1:
        return "medium"
    else:
        return "low"

def extract_sources_from_response(response) -> List[str]:
    """
    Extract source files from the response, handling different response types.
    """
    sources = []
    
    try:
        # Handle different response types
        if hasattr(response, 'source_nodes') and response.source_nodes:
            source_files = []
            for node in response.source_nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    file_name = node.metadata.get("file_name", "Unknown")
                    if file_name not in source_files:
                        source_files.append(file_name)
            sources = sorted(source_files)
        
        # Fallback for custom engines
        elif hasattr(response, 'metadata') and response.metadata:
            sources = response.metadata.get("sources", [])
        
        # If no sources found, provide default
        if not sources:
            sources = ["Budget Documents"]
            
    except Exception as e:
        logger.warning(f"Could not extract sources: {e}")
        sources = ["Budget Documents"]
    
    return sources

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the Hybrid Graph RAG engine with optional decomposition.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    import time
    start_time = time.time()
    
    # Determine query complexity
    complexity = determine_query_complexity(request.query)
    
    # Auto-select decomposition for high complexity queries if none specified
    decomposition_type = request.decomposition_type
    if decomposition_type == DecompositionType.none and complexity == "high":
        decomposition_type = DecompositionType.multistep
        logger.info(f"Auto-selected multistep decomposition for high complexity query")
    
    try:
        query_engine = get_query_engine(decomposition_type)
        
        # Process the query with retries
        response = None
        last_error = None
        
        for attempt in range(request.max_retries):
            try:
                logger.info(f"Processing query (attempt {attempt + 1}): {request.query}")
                
                # Handle custom engine differently
                if decomposition_type == DecompositionType.custom:
                    response = query_engine.query(request.query)
                else:
                    response = query_engine.query(request.query)
                break
                
            except Exception as e:
                last_error = e
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt == request.max_retries - 1:
                    raise e
                continue
        
        if response is None:
            raise last_error or Exception("Failed to get response")
        
        # Extract sources
        sources = extract_sources_from_response(response)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get response text
        answer = response.response if hasattr(response, 'response') else str(response)
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            decomposition_type=decomposition_type.value,
            query_complexity=complexity,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing query after {processing_time:.2f}s: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Query processing failed: {str(e)}"
        )

@app.get("/query/simple")
async def simple_query(
    q: str = Query(..., description="Simple query parameter"),
    decomposition: DecompositionType = Query(
        default=DecompositionType.none,
        description="Decomposition type"
    )
):
    """
    Simple GET endpoint for quick queries.
    """
    request = QueryRequest(query=q, decomposition_type=decomposition)
    return await process_query(request)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Enhanced health check that shows loaded engines.
    """
    engines_status = {}
    
    for engine_type in DecompositionType:
        if engine_type.value in _engines_cache:
            engines_status[engine_type.value] = "loaded"
        else:
            engines_status[engine_type.value] = "not_loaded"
    
    return HealthResponse(
        status="healthy",
        engines_loaded=engines_status
    )

@app.get("/engines/preload")
async def preload_engines():
    """
    Endpoint to preload all engine types for faster subsequent queries.
    """
    results = {}
    
    for engine_type in DecompositionType:
        try:
            get_query_engine(engine_type)
            results[engine_type.value] = "success"
        except Exception as e:
            results[engine_type.value] = f"failed: {str(e)}"
            logger.error(f"Failed to preload {engine_type.value} engine: {e}")
    
    return {"preload_results": results}

@app.get("/engines/info")
async def get_engines_info():
    """
    Get information about available decomposition types and their use cases.
    """
    return {
        "available_engines": {
            "none": {
                "description": "Direct query without decomposition",
                "best_for": "Simple, straightforward queries",
                "performance": "Fastest"
            },
            "multistep": {
                "description": "Sequential query decomposition",
                "best_for": "Complex analytical queries requiring step-by-step reasoning",
                "performance": "Moderate"
            },
            "subquestion": {
                "description": "Parallel sub-question processing",
                "best_for": "Queries with multiple independent components",
                "performance": "Moderate (parallel processing)"
            },
            "custom": {
                "description": "Custom decomposition logic with full control",
                "best_for": "Domain-specific complex queries with custom logic",
                "performance": "Variable"
            }
        },
        "auto_selection": {
            "description": "System automatically selects multistep for high complexity queries",
            "complexity_factors": [
                "compare", "analysis", "percentage", "change", "between",
                "calculate", "difference", "trend", "breakdown"
            ]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)