"""
Pydantic v2 Models for API Request & Response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Citation(BaseModel):
    """Grounding citation information for retrieved context snippets."""
    document_id: Optional[str] = Field(None, description="UUID of the parent document")
    filename: Optional[str] = Field(None, description="Original filename")
    page_number: Optional[int] = Field(None, description="1-indexed document page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    relevance_score: float = Field(..., description="Confidence score from re-ranker [0.0 - 1.0]")
    snippet: str = Field(..., description="Context passage snippet used for generation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_12345",
                "filename": "quantum_mechanics.pdf",
                "page_number": 4,
                "chunk_id": "doc_12345_p4_c2",
                "relevance_score": 0.942,
                "snippet": "Schrodinger's equation describes the wave function..."
            }
        }
    )


class QueryRequest(BaseModel):
    """User search and question answering query payload."""
    question: str = Field(..., min_length=2, description="The user's question or search query")
    filter_document_id: Optional[str] = Field(None, description="Optional document_id to constrain search to")
    top_k: int = Field(5, ge=1, le=20, description="Number of re-ranked context contexts to synthesize")
    stream: bool = Field(False, description="Whether to stream the LLM response tokens")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the net profit reported in Table 3 of the financial statement?",
                "filter_document_id": None,
                "top_k": 5,
                "stream": False
            }
        }
    )


class RetrievalMetrics(BaseModel):
    """Latency and pipeline execution metrics."""
    retrieval_latency_ms: float = Field(..., description="Dense + Sparse + RRF latency in milliseconds")
    rerank_latency_ms: float = Field(..., description="Cross-encoder re-ranking latency in milliseconds")
    generation_latency_ms: float = Field(..., description="LLM synthesis latency in milliseconds")
    total_latency_ms: float = Field(..., description="End-to-end request latency in milliseconds")
    candidates_retrieved: int = Field(..., description="Total candidates retrieved before re-ranking")


class QueryResponse(BaseModel):
    """Full question answering response with citations and execution telemetry."""
    question: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Synthesized answer grounded in document contexts")
    citations: List[Citation] = Field(default_factory=list, description="Grounding source citations")
    metrics: RetrievalMetrics = Field(..., description="Performance latency metrics")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the eigenvalues of matrix A?",
                "answer": "According to [Page 7, Doc: linear_algebra.pdf], the eigenvalues are $\\lambda_1 = 3$ and $\\lambda_2 = -1$.",
                "citations": [],
                "metrics": {
                    "retrieval_latency_ms": 45.2,
                    "rerank_latency_ms": 82.1,
                    "generation_latency_ms": 310.4,
                    "total_latency_ms": 437.7,
                    "candidates_retrieved": 25
                }
            }
        }
    )


class DocumentUploadResponse(BaseModel):
    """Response payload following document ingestion and indexing."""
    document_id: str = Field(..., description="Assigned unique document identifier")
    filename: str = Field(..., description="Uploaded document filename")
    total_pages: int = Field(..., description="Extracted page count")
    total_chunks: int = Field(..., description="Number of semantic chunks indexed")
    total_tables: int = Field(..., description="Number of markdown tables detected and preserved")
    has_scanned_pages: bool = Field(..., description="Whether OCR fallback was triggered")
    processing_time_ms: float = Field(..., description="Processing and indexing duration in ms")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "7fa8415c-99d8-4f24-9df2-5d9e52e42095",
                "filename": "annual_report_2024.pdf",
                "total_pages": 12,
                "total_chunks": 34,
                "total_tables": 6,
                "has_scanned_pages": False,
                "processing_time_ms": 820.5
            }
        }
    )


class HealthResponse(BaseModel):
    """System health check and vector index statistics."""
    status: str = Field(..., description="System operational status ('healthy' or 'degraded')")
    environment: str = Field(..., description="Application deployment environment")
    documents_indexed: int = Field(..., description="Total unique documents in vector store")
    total_chunks: int = Field(..., description="Total chunks in vector store")
    chunks_with_tables: int = Field(..., description="Chunks containing structured tables")
    chunks_with_math: int = Field(..., description="Chunks containing mathematical formulas")
    embedding_provider: str = Field(..., description="Configured embedding engine")
    embedding_model: str = Field(..., description="Active embedding model name")
    reranker_model: str = Field(..., description="Active cross-encoder re-ranking model")
    llm_model: str = Field(..., description="Active LLM model")
