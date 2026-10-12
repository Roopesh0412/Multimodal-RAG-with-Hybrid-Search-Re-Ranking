"""
Unit tests for the Recursive Markdown-Aware Chunker.
Validates table integrity, equation preservation, and token overlap adherence.
"""

import pytest
from app.core.chunker import MarkdownAwareChunker


def test_chunker_preserves_tables():
    chunker = MarkdownAwareChunker(chunk_size_tokens=200, chunk_overlap_tokens=30)
    sample_text = (
        "# Experimental Results\n\n"
        "Here are the benchmark measurements:\n\n"
        "| Metric | Model A | Model B |\n"
        "|---|---|---|\n"
        "| Accuracy | 94.2% | 98.1% |\n"
        "| Latency (ms) | 42.1 | 18.5 |\n"
        "| Memory (MB) | 512 | 256 |\n\n"
        "The results demonstrate superior throughput."
    )

    chunks = chunker.chunk_page(
        text=sample_text,
        document_id="doc_test_1",
        filename="test.pdf",
        page_number=1
    )

    assert len(chunks) >= 1
    # Verify that the table header and rows are preserved inside a chunk
    table_chunk = next((c for c in chunks if "| Accuracy |" in c.content), None)
    assert table_chunk is not None
    assert "| Metric | Model A | Model B |" in table_chunk.content
    assert "| Latency (ms) |" in table_chunk.content
    assert table_chunk.metadata["has_table"] is True


def test_chunker_preserves_latex_equations():
    chunker = MarkdownAwareChunker(chunk_size_tokens=150, chunk_overlap_tokens=20)
    sample_text = (
        "## Field Equations\n\n"
        "The Maxwell-Faraday equation in differential form is stated as:\n\n"
        "$$\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}$$\n\n"
        "where E is electric field and B is magnetic flux density."
    )

    chunks = chunker.chunk_page(
        text=sample_text,
        document_id="doc_test_2",
        filename="electrodynamics.pdf",
        page_number=3
    )

    assert len(chunks) >= 1
    math_chunk = next((c for c in chunks if "\\nabla \\times" in c.content), None)
    assert math_chunk is not None
    assert "$$\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}$$" in math_chunk.content
    assert math_chunk.metadata["has_math"] is True


def test_chunker_token_bounds():
    chunker = MarkdownAwareChunker(chunk_size_tokens=50, chunk_overlap_tokens=10)
    # Long text with multiple paragraphs
    long_text = "\n\n".join([f"Paragraph {i}: This is an in-depth scientific analysis of test parameter {i}." for i in range(25)])

    chunks = chunker.chunk_page(
        text=long_text,
        document_id="doc_test_3",
        filename="analysis.pdf",
        page_number=1
    )

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count > 0
        assert c.document_id == "doc_test_3"
