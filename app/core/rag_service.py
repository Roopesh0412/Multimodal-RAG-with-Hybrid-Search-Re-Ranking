"""
Unified Enterprise RAG Pipeline Orchestrator.
Coordinates ingestion, chunking, embedding, persistent storage, hybrid retrieval,
re-ranking, and response generation.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

from app.config import Settings, get_settings
from app.core.document_processor import DocumentProcessor, DocumentExtractionResult
from app.core.chunker import MarkdownAwareChunker, DocumentChunk
from app.core.embeddings import get_embedding_service, BaseEmbeddingService
from app.core.vector_store import ChromaVectorStore
from app.core.hybrid_retriever import HybridRetriever, HybridCandidate
from app.core.reranker import Reranker, RerankedContext
from app.core.generator import LLMGenerator

logger = logging.getLogger("rag_enterprise.service")


class RAGService:
    """
    Enterprise RAG Service encapsulating the entire multimodal workflow.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # Ingestion & Chunking
        self.processor = DocumentProcessor(
            min_text_threshold=self.settings.OCR_MIN_TEXT_THRESHOLD,
            enable_ocr=self.settings.ENABLE_OCR_FALLBACK
        )
        self.chunker = MarkdownAwareChunker(
            chunk_size_tokens=self.settings.CHUNK_SIZE_TOKENS,
            chunk_overlap_tokens=self.settings.CHUNK_OVERLAP_TOKENS
        )

        # Embeddings & Vector Storage
        self.embedding_service: BaseEmbeddingService = get_embedding_service(
            provider=self.settings.EMBEDDING_PROVIDER,
            model_name=self.settings.EMBEDDING_MODEL_NAME,
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.LLM_BASE_URL
        )
        self.vector_store = ChromaVectorStore(
            persist_directory=self.settings.chroma_path
        )

        # Hybrid Retrieval & Re-ranking
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            dense_top_k=self.settings.DENSE_TOP_K,
            sparse_top_k=self.settings.SPARSE_TOP_K,
            rrf_k=self.settings.RRF_K
        )
        self.reranker = Reranker(
            model_name=self.settings.RERANKER_MODEL_NAME,
            top_k=self.settings.RERANKER_TOP_K
        )

        # LLM Synthesis
        self.generator = LLMGenerator(
            provider=self.settings.LLM_PROVIDER,
            model=self.settings.LLM_MODEL,
            base_url=self.settings.LLM_BASE_URL,
            api_key=self.settings.OPENAI_API_KEY or self.settings.GROQ_API_KEY,
            temperature=self.settings.LLM_TEMPERATURE,
            max_tokens=self.settings.LLM_MAX_TOKENS
        )

        logger.info("RAGService fully initialized and ready.")

    def index_document(self, file_path: Path, document_id: Optional[str] = None) -> Tuple[DocumentExtractionResult, List[DocumentChunk], float]:
        """
        Processes a PDF document, recursively chunks its content, embeds chunks,
        and saves them into ChromaDB and the BM25 index.
        """
        start_time = time.perf_counter()
        doc_id = document_id or str(uuid.uuid4())

        # 1. Layout Extraction & Markdown Table Parsing
        extraction_result = self.processor.process_pdf(file_path, doc_id)

        # 2. Markdown-Aware Recursive Chunking
        all_chunks: List[DocumentChunk] = []
        chunk_idx = 0
        for page in extraction_result.pages:
            page_chunks = self.chunker.chunk_page(
                text=page.text_content,
                document_id=doc_id,
                filename=file_path.name,
                page_number=page.page_number,
                start_chunk_index=chunk_idx
            )
            all_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        if not all_chunks:
            logger.warning(f"No chunks produced for document {file_path.name}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return extraction_result, [], duration_ms

        # 3. Dense Embeddings Generation
        chunk_texts = [c.content for c in all_chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # 4. Storage & BM25 Invalidation
        self.vector_store.add_chunks(all_chunks, embeddings)
        self.retriever.mark_dirty()

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Successfully indexed '{file_path.name}': {len(all_chunks)} chunks, "
            f"{extraction_result.total_tables} tables in {duration_ms:.1f}ms."
        )

        return extraction_result, all_chunks, duration_ms

    def query(
        self,
        question: str,
        filter_document_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Executes end-to-end question answering pipeline:
        Hybrid Search (Dense + BM25 + RRF) -> Cross-Encoder Re-ranking -> LLM Synthesis.
        Measures granular latency benchmarks (<400ms target retrieval).
        """
        t_start = time.perf_counter()

        # Step 1: Hybrid Retrieval (Dense + Sparse + RRF)
        t_ret_start = time.perf_counter()
        candidates: List[HybridCandidate] = self.retriever.retrieve(
            query=question,
            filter_document_id=filter_document_id,
            top_candidates_limit=25
        )
        t_ret_end = time.perf_counter()
        retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000

        # Step 2: Cross-Encoder Re-Ranking
        t_rerank_start = time.perf_counter()
        reranked_contexts: List[RerankedContext] = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=top_k
        )
        t_rerank_end = time.perf_counter()
        rerank_latency_ms = (t_rerank_end - t_rerank_start) * 1000

        # Step 3: LLM Generation
        t_gen_start = time.perf_counter()
        answer, citations = self.generator.generate(
            query=question,
            contexts=reranked_contexts
        )
        t_gen_end = time.perf_counter()
        generation_latency_ms = (t_gen_end - t_gen_start) * 1000

        total_latency_ms = (t_gen_end - t_start) * 1000

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "metrics": {
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "rerank_latency_ms": round(rerank_latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "candidates_retrieved": len(candidates)
            },
            "reranked_contexts": reranked_contexts
        }

    async def query_stream(
        self,
        question: str,
        filter_document_id: Optional[str] = None,
        top_k: int = 5
    ) -> Tuple[AsyncGenerator[str, None], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes hybrid retrieval and re-ranking, and prepares async stream for generation.
        """
        t_ret_start = time.perf_counter()
        candidates = self.retriever.retrieve(
            query=question,
            filter_document_id=filter_document_id,
            top_candidates_limit=25
        )
        t_ret_end = time.perf_counter()
        retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000

        t_rerank_start = time.perf_counter()
        reranked_contexts = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=top_k
        )
        t_rerank_end = time.perf_counter()
        rerank_latency_ms = (t_rerank_end - t_rerank_start) * 1000

        citations = [
            {
                "document_id": ctx.metadata.get("document_id"),
                "filename": ctx.metadata.get("filename"),
                "page_number": ctx.metadata.get("page_number"),
                "chunk_id": ctx.chunk_id,
                "relevance_score": round(ctx.rerank_score, 4),
                "snippet": ctx.content[:300] + ("..." if len(ctx.content) > 300 else "")
            }
            for ctx in reranked_contexts
        ]

        metrics = {
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "rerank_latency_ms": round(rerank_latency_ms, 2),
            "candidates_retrieved": len(candidates)
        }

        stream = self.generator.stream_generate(question, reranked_contexts)
        return stream, metrics, citations
