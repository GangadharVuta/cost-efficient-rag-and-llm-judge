import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.config import settings
from src.rag.pipeline import RAGPipeline

app = FastAPI(
    title="Cost-Efficient RAG Application API",
    description="Low-cost grounded QA RAG service with telemetry and citation tracking",
    version="1.0.0"
)

# Global pipeline instance
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is vector database cost scaling?")
    top_k: Optional[int] = Field(default=4, ge=1, le=20)
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None)

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    used_chunks_count: int
    telemetry: Dict[str, Any]

class IngestResponse(BaseModel):
    status: str
    file_path: Optional[str] = None
    doc_id: Optional[str] = None
    chunks_added: int
    latency_ms: float

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store_type": rag_pipeline.store_type,
        "total_vectors_stored": rag_pipeline.store.count(),
        "embedding_model": settings.rag.embedding_model,
        "llm_provider": settings.llm.provider
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    res = rag_pipeline.query(
        query_text=request.query,
        top_k=request.top_k or 4,
        metadata_filter=request.metadata_filter
    )
    return res

@app.post("/ingest", response_model=IngestResponse)
def ingest_file_path(file_path: str = Form(...), category: str = Form("general")):
    try:
        res = rag_pipeline.ingest_file(file_path=file_path, category=category)
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
def clear_vectors():
    rag_pipeline.store.clear()
    return {"status": "success", "message": "Vector store cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
