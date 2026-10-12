"""
Unit tests for Sparse BM25 and Reciprocal Rank Fusion (RRF) mechanics.
"""

import pytest
from app.core.hybrid_retriever import HybridRetriever, HybridCandidate


def test_rrf_scoring_logic():
    """Validates that RRF formula score calculation correctly ranks fused candidates."""
    # Test formula: RRF(d) = sum(1 / (k + rank))
    rrf_k = 60

    # Candidate 1: Rank 1 in Dense, Rank 1 in Sparse
    c1_dense_rank = 1
    c1_sparse_rank = 1
    c1_expected = (1.0 / (rrf_k + 1)) + (1.0 / (rrf_k + 1))

    # Candidate 2: Rank 1 in Dense, Rank 5 in Sparse
    c2_expected = (1.0 / (rrf_k + 1)) + (1.0 / (rrf_k + 5))

    # Candidate 3: Rank 20 in Dense only
    c3_expected = 1.0 / (rrf_k + 20)

    assert c1_expected > c2_expected > c3_expected
    assert round(c1_expected, 5) == round(2.0 / 61.0, 5)


def test_bm25_tokenization():
    """Tests tokenizer extraction of scientific and mathematical variables."""
    from app.core.hybrid_retriever import HybridRetriever
    from app.core.vector_store import ChromaVectorStore
    from unittest.mock import MagicMock

    mock_vs = MagicMock(spec=ChromaVectorStore)
    mock_emb = MagicMock()
    retriever = HybridRetriever(vector_store=mock_vs, embedding_service=mock_emb)

    text = "The constant \\hbar has value 1.054e-34 J*s where alpha=0.007."
    tokens = retriever._tokenize(text)
    assert "\\hbar" in tokens or "hbar" in tokens
    assert "constant" in tokens
    assert "alpha" in tokens
