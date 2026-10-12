"""
FastAPI Routes implementation for Document Ingestion, Hybrid Search, and Health.
"""

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query as FastQuery, status
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.api.models import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    HealthResponse,
    Citation,
    RetrievalMetrics
)
from app.core.rag_service import RAGService

logger = logging.getLogger("rag_enterprise.api")

router = APIRouter(prefix="/api/v1", tags=["Multimodal Enterprise RAG"])

# Singleton RAGService holder
_rag_service_instance: Optional[RAGService] = None


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    """Dependency injector providing RAGService instance."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService(settings)
    return _rag_service_instance


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Index PDF Document",
    description="Uploads a complex scientific or tabular PDF, parses tables into Markdown, executes OCR fallback if needed, chunks recursively, and indexes into ChromaDB."
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to index"),
    rag_service: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings)
):
    # Validate file format
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only PDF (.pdf) files are supported."
        )

    document_id = str(uuid.uuid4())
    upload_dir = settings.upload_path
    saved_file_path = upload_dir / f"{document_id}_{file.filename}"

    try:
        # Save file to disk
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Index document through RAG service
        extraction_result, chunks, duration_ms = rag_service.index_document(
            file_path=saved_file_path,
            document_id=document_id
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_pages=extraction_result.total_pages,
            total_chunks=len(chunks),
            total_tables=extraction_result.total_tables,
            has_scanned_pages=extraction_result.has_scanned_pages,
            processing_time_ms=round(duration_ms, 2)
        )
    except Exception as e:
        logger.error(f"Error processing upload for {file.filename}: {e}", exc_info=True)
        # Cleanup uploaded file on catastrophic error
        if saved_file_path.exists():
            saved_file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {str(e)}"
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query with Hybrid Search & Re-ranking",
    description="Executes Dense + BM25 + Reciprocal Rank Fusion, followed by Cross-Encoder re-ranking and LLM generation with citations."
)
async def query_documents(
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    if payload.stream:
        # Return real-time streaming response
        async def event_generator():
            stream, metrics, citations = await rag_service.query_stream(
                question=payload.question,
                filter_document_id=payload.filter_document_id,
                top_k=payload.top_k
            )
            # Yield token chunks
            async for token in stream:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Yield citations and metrics metadata at completion
            yield f"data: {json.dumps({'type': 'metadata', 'citations': citations, 'metrics': metrics})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    # Synchronous query execution
    try:
        result = rag_service.query(
            question=payload.question,
            filter_document_id=payload.filter_document_id,
            top_k=payload.top_k
        )

        citations_models = [Citation(**c) for c in result["citations"]]
        metrics_model = RetrievalMetrics(**result["metrics"])

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            citations=citations_models,
            metrics=metrics_model
        )
    except Exception as e:
        logger.error(f"Query execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Metrics",
    description="Returns API status, persistent index volume, tables/math statistics, and active model providers."
)
async def health_check(
    rag_service: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings)
):
    try:
        stats = rag_service.vector_store.get_stats()
        return HealthResponse(
            status="healthy",
            environment=settings.ENVIRONMENT,
            documents_indexed=stats["total_documents"],
            total_chunks=stats["total_chunks"],
            chunks_with_tables=stats["chunks_with_tables"],
            chunks_with_math=stats["chunks_with_math"],
            embedding_provider=settings.EMBEDDING_PROVIDER,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            reranker_model=settings.RERANKER_MODEL_NAME,
            llm_model=settings.LLM_MODEL
        )
    except Exception as e:
        logger.error(f"Health check inspection failed: {e}", exc_info=True)
        return HealthResponse(
            status="degraded",
            environment=settings.ENVIRONMENT,
            documents_indexed=0,
            total_chunks=0,
            chunks_with_tables=0,
            chunks_with_math=0,
            embedding_provider=settings.EMBEDDING_PROVIDER,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            reranker_model=settings.RERANKER_MODEL_NAME,
            llm_model=settings.LLM_MODEL
        )


@router.get(
    "/documents",
    summary="List Indexed Documents",
    description="Returns metadata of all indexed documents in the vector store."
)
async def list_documents(rag_service: RAGService = Depends(get_rag_service)):
    stats = rag_service.vector_store.get_stats()
    return {
        "total_documents": stats["total_documents"],
        "document_ids": stats["documents_indexed"],
        "total_chunks": stats["total_chunks"]
    }


@router.delete(
    "/documents/{document_id}",
    summary="Delete Document from Index",
    description="Removes all chunks of a document from ChromaDB and invalidates BM25 index."
)
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    deleted_count = rag_service.vector_store.delete_document(document_id)
    rag_service.retriever.mark_dirty()
    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document ID '{document_id}' not found."
        )
    return {
        "message": f"Successfully deleted document '{document_id}'",
        "chunks_removed": deleted_count
    }
