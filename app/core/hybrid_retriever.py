"""
Hybrid Retrieval Engine combining Dense Semantic Search and Sparse Lexical BM25 Search.
Merges candidate pools via Reciprocal Rank Fusion (RRF).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.core.embeddings import BaseEmbeddingService
from app.core.vector_store import ChromaVectorStore

logger = logging.getLogger("rag_enterprise.retriever")


@dataclass
class HybridCandidate:
    """Represents a fused candidate retrieved through hybrid search."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class HybridRetriever:
    """
    Hybrid Retriever executing concurrent Dense (ChromaDB) and Sparse (BM25Okapi)
    search followed by Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: BaseEmbeddingService,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k

        # BM25 In-Memory Index cache
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_chunks: List[Dict[str, Any]] = []
        self._is_bm25_dirty = True

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 with preservation of alphanumeric and math variables."""
        return re.findall(r'[A-Za-z0-9_]+|[\\$+*/^=<>~-]+', text.lower())

    def rebuild_bm25_index(self, document_id: Optional[str] = None):
        """Reconstructs BM25 lexical index from vector store chunks."""
        chunks = self.vector_store.get_all_chunks(document_id=document_id)
        self._bm25_chunks = chunks

        if not chunks:
            self._bm25_index = None
            self._is_bm25_dirty = False
            return

        corpus_tokenized = [self._tokenize(c["content"]) for c in chunks]
        self._bm25_index = BM25Okapi(corpus_tokenized)
        self._is_bm25_dirty = False
        logger.info(f"Rebuilt BM25 index with {len(chunks)} document chunks.")

    def _ensure_bm25_index(self):
        """Ensures the BM25 index is up-to-date."""
        if self._is_bm25_dirty or self._bm25_index is None:
            self.rebuild_bm25_index()

    def mark_dirty(self):
        """Flags that newly indexed documents require BM25 index rebuild."""
        self._is_bm25_dirty = True

    def search_sparse(
        self,
        query: str,
        top_k: int = 20,
        filter_document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes sparse lexical search via BM25Okapi.
        """
        self._ensure_bm25_index()
        if not self._bm25_index or not self._bm25_chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)

        # Pair scores with chunk objects and sort descending
        scored_candidates = []
        for idx, score in enumerate(scores):
            chunk = self._bm25_chunks[idx]
            if filter_document_id:
                if chunk["metadata"].get("document_id") != filter_document_id:
                    continue
            if score > 0.0:  # Only consider chunks with lexical overlap
                scored_candidates.append((score, chunk))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        results = []
        for rank, (score, chunk) in enumerate(top_candidates, start=1):
            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "sparse_score": float(score),
                "sparse_rank": rank
            })

        return results

    def search_dense(
        self,
        query: str,
        top_k: int = 20,
        filter_document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes dense semantic search via ChromaDB vector embeddings.
        """
        query_vector = self.embedding_service.embed_query(query)
        where_filter = {"document_id": filter_document_id} if filter_document_id else None

        dense_matches = self.vector_store.search_dense(
            query_embedding=query_vector,
            top_k=top_k,
            where_filter=where_filter
        )

        for rank, match in enumerate(dense_matches, start=1):
            match["dense_rank"] = rank

        return dense_matches

    def retrieve(
        self,
        query: str,
        filter_document_id: Optional[str] = None,
        top_candidates_limit: int = 25
    ) -> List[HybridCandidate]:
        """
        Executes Hybrid Retrieval combining Dense + Sparse search,
        fusing candidates via Reciprocal Rank Fusion (RRF).
        RRF Formula: RRF(d) = sum_{m in {dense, sparse}} 1 / (rrf_k + rank_m(d))
        """
        logger.debug(f"Executing hybrid retrieval for query: '{query}'")

        # 1. Dense Semantic Search
        dense_results = self.search_dense(
            query=query,
            top_k=self.dense_top_k,
            filter_document_id=filter_document_id
        )

        # 2. Sparse Lexical Search (BM25)
        sparse_results = self.search_sparse(
            query=query,
            top_k=self.sparse_top_k,
            filter_document_id=filter_document_id
        )

        # 3. Reciprocal Rank Fusion (RRF)
        candidate_pool: Dict[str, Dict[str, Any]] = {}

        # Process Dense results
        for item in dense_results:
            cid = item["chunk_id"]
            dense_rank = item["dense_rank"]
            rrf_contribution = 1.0 / (self.rrf_k + dense_rank)

            candidate_pool[cid] = {
                "chunk_id": cid,
                "content": item["content"],
                "metadata": item["metadata"],
                "rrf_score": rrf_contribution,
                "dense_rank": dense_rank,
                "dense_score": item.get("dense_score"),
                "sparse_rank": None,
                "sparse_score": None
            }

        # Process Sparse results
        for item in sparse_results:
            cid = item["chunk_id"]
            sparse_rank = item["sparse_rank"]
            rrf_contribution = 1.0 / (self.rrf_k + sparse_rank)

            if cid in candidate_pool:
                candidate_pool[cid]["rrf_score"] += rrf_contribution
                candidate_pool[cid]["sparse_rank"] = sparse_rank
                candidate_pool[cid]["sparse_score"] = item.get("sparse_score")
            else:
                candidate_pool[cid] = {
                    "chunk_id": cid,
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": rrf_contribution,
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": sparse_rank,
                    "sparse_score": item.get("sparse_score")
                }

        # Sort combined candidate pool by RRF score descending
        sorted_candidates = sorted(
            candidate_pool.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        fused_results = [
            HybridCandidate(
                chunk_id=c["chunk_id"],
                content=c["content"],
                metadata=c["metadata"],
                rrf_score=c["rrf_score"],
                dense_rank=c["dense_rank"],
                sparse_rank=c["sparse_rank"],
                dense_score=c["dense_score"],
                sparse_score=c["sparse_score"]
            )
            for c in sorted_candidates[:top_candidates_limit]
        ]

        logger.info(
            f"Hybrid retrieval produced {len(fused_results)} candidates "
            f"(Dense: {len(dense_results)}, Sparse: {len(sparse_results)})"
        )
        return fused_results
