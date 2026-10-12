"""
Cross-Encoder Re-Ranking Engine.
Evaluates query-context relevance with deep attention interaction.
Supports sentence-transformers CrossEncoder ('cross-encoder/ms-marco-MiniLM-L-6-v2')
and FlashRank for ultra-fast CPU inference.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from app.core.hybrid_retriever import HybridCandidate

logger = logging.getLogger("rag_enterprise.reranker")


@dataclass
class RerankedContext:
    """Represents a final high-confidence context snippet after re-ranking."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    rerank_score: float
    rrf_score: float
    original_rank: int
    final_rank: int


class Reranker:
    """
    Two-stage re-ranking component using Cross-Encoder architectures.
    Returns the top-K highest-confidence contexts.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._cross_encoder = None
        self._flashrank_client = None
        self._backend = "cross_encoder"  # or "flashrank"

    def _init_model(self):
        """Initializes CrossEncoder or FlashRank."""
        if self._cross_encoder is None and self._flashrank_client is None:
            # Try sentence-transformers CrossEncoder
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.model_name)
                self._backend = "cross_encoder"
                logger.info(f"Loaded CrossEncoder model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}). Attempting FlashRank.")

            # Try FlashRank fallback
            try:
                from flashrank import Ranker
                self._flashrank_client = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
                self._backend = "flashrank"
                logger.info("Loaded FlashRank Ranker model (ms-marco-TinyBERT-L-2-v2).")
                return
            except Exception as e2:
                logger.warning(f"FlashRank unavailable: {e2}. Fallback to RRF rank ordering.")
                self._backend = "passthrough"

    def _sigmoid(self, x: float) -> float:
        """Applies sigmoid to logits to produce probability [0, 1]."""
        return 1.0 / (1.0 + np.exp(-x))

    def rerank(
        self,
        query: str,
        candidates: List[HybridCandidate],
        top_k: Optional[int] = None
    ) -> List[RerankedContext]:
        """
        Re-ranks hybrid candidate pool and returns top-K contexts.
        """
        if not candidates:
            return []

        effective_top_k = top_k or self.top_k
        self._init_model()

        # Handle passthrough if no re-ranking model is operational
        if self._backend == "passthrough":
            sorted_candidates = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
            results = []
            for rank, c in enumerate(sorted_candidates[:effective_top_k], start=1):
                results.append(
                    RerankedContext(
                        chunk_id=c.chunk_id,
                        content=c.content,
                        metadata=c.metadata,
                        rerank_score=c.rrf_score,
                        rrf_score=c.rrf_score,
                        original_rank=rank,
                        final_rank=rank
                    )
                )
            return results

        # 1. CrossEncoder inference
        if self._backend == "cross_encoder":
            pairs = [[query, c.content] for c in candidates]
            raw_scores = self._cross_encoder.predict(pairs)

            scored_candidates = []
            for orig_rank, (c, score) in enumerate(zip(candidates, raw_scores), start=1):
                prob_score = float(self._sigmoid(score))
                scored_candidates.append({
                    "candidate": c,
                    "score": prob_score,
                    "orig_rank": orig_rank
                })

            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_items = scored_candidates[:effective_top_k]

            reranked_results = []
            for final_rank, item in enumerate(top_items, start=1):
                c = item["candidate"]
                reranked_results.append(
                    RerankedContext(
                        chunk_id=c.chunk_id,
                        content=c.content,
                        metadata=c.metadata,
                        rerank_score=item["score"],
                        rrf_score=c.rrf_score,
                        original_rank=item["orig_rank"],
                        final_rank=final_rank
                    )
                )
            return reranked_results

        # 2. FlashRank inference
        if self._backend == "flashrank":
            from flashrank import RerankRequest
            passages = [{"id": c.chunk_id, "text": c.content, "candidate": c} for c in candidates]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self._flashrank_client.rerank(rerank_request)

            reranked_results = []
            for final_rank, item in enumerate(results[:effective_top_k], start=1):
                # FlashRank returns dict with 'id', 'text', 'score'
                orig_candidate = next(c for c in candidates if c.chunk_id == item["id"])
                reranked_results.append(
                    RerankedContext(
                        chunk_id=item["id"],
                        content=item["text"],
                        metadata=orig_candidate.metadata,
                        rerank_score=float(item["score"]),
                        rrf_score=orig_candidate.rrf_score,
                        original_rank=1,
                        final_rank=final_rank
                    )
                )
            return reranked_results

        return []
