# Hybrid Graph RAG QA

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-blueviolet)]()
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)]()

## 📌 Overview
**Hybrid Graph RAG QA** is a production-ready **Retrieval-Augmented Generation (RAG)** system designed to answer **complex questions from budget documents** (e.g., government reports).  

It combines:
- **Knowledge Graph (Neo4j)** → Structured facts & explicit relationships.  
- **Vector Index (Semantic Search)** → Contextual understanding of unstructured text.  

This hybrid approach enables **precise factual answers** *and* **context-rich analytical insights**, with sources cited for verifiability.

---

## ⚙️ Features
✅ Multi-strategy **PDF parsing** (Docling, OCR, PyMuPDF, fallback readers).  
✅ **Semantic splitting** & **metadata enrichment** (year, department, budget amounts).  
✅ **Knowledge Graph** construction with predefined schema.  
✅ **Hybrid Vector Indexing** (financial, departmental, full index).  
✅ **Hybrid retrieval pipeline**: *Vector-First, Graph-Expansion*.  
✅ **Multi-step & sub-question decomposition** for complex queries.  
✅ Multi-query decomposition strategies:

     - Multistep Engine → Sequential, dependent queries.
     - Subquestion Engine → Parallel, comparative queries.
     ✅ Automatic query complexity detection (low, medium, high).
     ✅ Adaptive orchestration (auto-selects decomposition strategy).
     ✅ On-demand query engine caching → Optimized memory use.
     ✅ Expert financial analyst synthesis prompt → Guides reasoning & ensures citations.
     ✅ FastAPI REST API with multiple endpoints.
     ✅ Dockerized deployment for production.

---

## 🏗️ System Architecture

### 🔹 Data Ingestion Pipeline
Orchestrated by `run_ingestion.py`:
1. **Document Parsing** (`document_parser.py`)  
   - Multi-parser strategy (Docling, OCR, PyMuPDF, fallback).  
   - Intelligent chunking (semantic or sentence-based).  
   - Metadata enrichment (year, department, budget amount).  

2. **Knowledge Graph Construction** (`graph_builder.py`)  
   - Predefined schema: `DEPARTMENT`, `FISCAL_YEAR`, `PROGRAM`.  
   - Multi-extractor entity & relationship detection.  
   - Post-processing with Cypher queries (hierarchies, cross-year links).  

3. **Vector Index Construction** (`vectorizer.py`)  
   - Metadata cleaning & adaptive chunking.  
   - Hybrid indices (`full`, `financial`, `departmental`).  

---

### 🔹 Retrieval Pipeline
Managed by `engine.py` & `retrievers.py`:
1. 🌊 **Vector Search** → Retrieve broad semantic context.  
2. 🔗 **Entity Linking** → Extract factual entities from text.  
3. 🕸 **Graph Traversal** → Enrich context with structured facts.  

**Query Engines**:  
- `create_base_query_engine` → Default hybrid retriever.  
- `create_multistep_query_engine` → For sequential analysis.  
- `create_subquestion_query_engine` → For comparative queries.  

All answers use an **expert financial analyst synthesis prompt**, with **citations from source PDFs**.  

---

### 🔹 API & Orchestration
Implemented in `api/main.py` with **FastAPI**.  

**Endpoints**:  
- `POST /query` → Submit complex queries (JSON body).  
- `GET /query/simple` → Quick lookups.  
- `GET /health` → Health check.  
- `GET /engines/preload` → Preload query engines.  
- `GET /engines/info` → Engine discovery.  

**Features**:  
- On-demand Engine Loading & Caching → Engines created only when needed.
- Automatic query complexity detection.  
- Smart routing (simple vs multistep vs sub-question).  
- Clean response format (answer, sources, processing time).  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/hybrid-graph-rag-qa.git

cd hybrid-graph-rag-qa
```

### 2️⃣ Build & Run with Docker
```bash
docker compose up --build --force-recreate -d
```

### 3️⃣ Run API Locally 
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📂 Ingestion Pipeline Usage

### Enhanced Strategy (default)
```bash
python run_ingestion.py
```

### Simple Strategy (fallback)
```bash
python run_ingestion.py --vector-strategy simple
```

### Hybrid Strategy (multiple indices)
```bash
python run_ingestion.py --vector-strategy hybrid
```

### Custom Data Path
```bash
python run_ingestion.py --data-path /path/to/your/pdfs
```

---

## 🔍 Querying the System

### Example Query
```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "How the education budget in 2024 differs from 2023?"}'
```

---

## 🛠 Tech Stack
- **Python 3.10+**  
- **FastAPI** (REST API)  
- **Neo4j** (Knowledge Graph DB)  
- **Vector DB** (for embeddings)  
- **LlamaIndex** (parsing, indexing, retrieval)  
- **Docker & Docker Compose**  

---

## 📊 Example Workflow
1. Place budget PDFs in `./data/pdfs/`.  
2. Run ingestion → `python run_ingestion.py`.  
3. Start API → `uvicorn api.main:app --reload`.  
4. Query with `curl` or API client.  

---

## 📁 Project Structure
```
hybrid-graph-rag-qa/
│── api/
│   ├── main.py               # FastAPI entrypoint
│── ingestion/
│   ├── run_ingestion.py      # Orchestrator script
│   ├── document_parser.py    # PDF parsing & enrichment
│   ├── graph_builder.py      # Knowledge Graph builder
│   ├── vectorizer.py         # Vector index creation
│── retrievers/
│   ├── retrievers.py         # Hybrid retriever logic
│── engine/
│   ├── engine.py             # Query engine factory
│── data/
│   ├── pdfs/                 # Input PDF documents
│── docker-compose.yml        # Docker services
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
```

---


