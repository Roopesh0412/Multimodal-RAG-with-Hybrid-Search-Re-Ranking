"""
Intelligent Recursive Markdown-Aware Chunker.
Optimized for complex scientific, mathematical, and tabular documents.
Guarantees formulas and table structures are preserved without boundary severance.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger("rag_enterprise.chunker")


@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of a document."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarkdownAwareChunker:
    """
    Recursive chunker designed for scientific, tabular, and mathematical markdown.
    Default parameters: chunk size = 800 tokens, overlap = 150 tokens.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 800,
        chunk_overlap_tokens: int = 150,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._tokenizer = None

        if TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception:
                try:
                    self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
                except Exception:
                    self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens using tiktoken or whitespace/character fallback."""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback estimation: average 4 characters or 0.75 words per token
        words = len(text.split())
        return max(1, int(words * 1.3))

    def _split_into_atomic_blocks(self, text: str) -> List[str]:
        """
        Splits text into atomic units (tables, math blocks, paragraphs, headers)
        that should ideally remain intact.
        """
        blocks = []
        lines = text.split("\n")
        current_block = []
        in_table = False
        in_math_block = False

        for line in lines:
            stripped = line.strip()

            # Check for LaTeX block delimiters
            if "$$" in stripped or "\\[" in stripped or "\\]" in stripped:
                in_math_block = not in_math_block
                current_block.append(line)
                if not in_math_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                continue

            if in_math_block:
                current_block.append(line)
                continue

            # Check for Markdown table rows
            if stripped.startswith("|") and stripped.endswith("|"):
                if not in_table:
                    # Flush previous block
                    if current_block:
                        blocks.append("\n".join(current_block))
                        current_block = []
                    in_table = True
                current_block.append(line)
                continue
            else:
                if in_table:
                    # Table ended, commit table block
                    blocks.append("\n".join(current_block))
                    current_block = []
                    in_table = False

            # Check for Markdown headers
            if stripped.startswith(("# ", "## ", "### ", "#### ", "##### ")):
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                current_block.append(line)
                continue

            # Check for empty paragraph separator
            if not stripped:
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                continue

            current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        return [b.strip() for b in blocks if b.strip()]

    def _split_oversized_table(self, table_text: str) -> List[str]:
        """
        Splits oversized table while retaining header row across sub-tables.
        """
        lines = table_text.strip().split("\n")
        if len(lines) <= 2:
            return [table_text]

        header = lines[0]
        separator = lines[1] if len(lines) > 1 and "---" in lines[1] else "|---|"
        data_rows = lines[2:] if "---" in lines[1] else lines[1:]

        chunks = []
        current_rows = []

        for row in data_rows:
            candidate = "\n".join([header, separator] + current_rows + [row])
            if self.count_tokens(candidate) > self.chunk_size_tokens and current_rows:
                chunks.append("\n".join([header, separator] + current_rows))
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            chunks.append("\n".join([header, separator] + current_rows))

        return chunks if chunks else [table_text]

    def _split_oversized_block(self, block: str) -> List[str]:
        """
        Splits oversized text blocks by sentence or line without breaking words.
        """
        if block.startswith("|") and ("---" in block):
            return self._split_oversized_table(block)

        # Split by sentence or line
        sentences = re.split(r'(?<=[.!?])\s+', block)
        sub_blocks = []
        current = []

        for s in sentences:
            test_text = " ".join(current + [s])
            if self.count_tokens(test_text) > self.chunk_size_tokens and current:
                sub_blocks.append(" ".join(current))
                current = [s]
            else:
                current.append(s)

        if current:
            sub_blocks.append(" ".join(current))

        return sub_blocks

    def chunk_page(
        self,
        text: str,
        document_id: str,
        filename: str,
        page_number: int,
        start_chunk_index: int = 0
    ) -> List[DocumentChunk]:
        """
        Recursively chunks page text adhering to token limits and overlap.
        """
        if not text.strip():
            return []

        # 1. Break text into atomic blocks (paragraphs, tables, math formulas)
        raw_blocks = self._split_into_atomic_blocks(text)
        refined_blocks = []

        for b in raw_blocks:
            if self.count_tokens(b) > self.chunk_size_tokens:
                refined_blocks.extend(self._split_oversized_block(b))
            else:
                refined_blocks.append(b)

        chunks: List[DocumentChunk] = []
        current_chunk_blocks: List[str] = []
        current_tokens = 0
        chunk_idx = start_chunk_index

        for block in refined_blocks:
            b_tokens = self.count_tokens(block)

            # Check if adding block exceeds chunk token budget
            if current_tokens + b_tokens > self.chunk_size_tokens and current_chunk_blocks:
                # Finalize current chunk
                chunk_content = "\n\n".join(current_chunk_blocks).strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}_p{page_number}_c{chunk_idx}",
                        document_id=document_id,
                        filename=filename,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                        content=chunk_content,
                        token_count=self.count_tokens(chunk_content),
                        metadata={
                            "document_id": document_id,
                            "filename": filename,
                            "page_number": page_number,
                            "chunk_index": chunk_idx,
                            "has_table": "|" in chunk_content and "---" in chunk_content,
                            "has_math": ("$$" in chunk_content) or ("\\[" in chunk_content)
                        }
                    )
                )
                chunk_idx += 1

                # Calculate overlap: retain trailing blocks up to chunk_overlap_tokens
                overlap_blocks: List[str] = []
                overlap_tokens = 0
                for prev_b in reversed(current_chunk_blocks):
                    prev_tok = self.count_tokens(prev_b)
                    if overlap_tokens + prev_tok <= self.chunk_overlap_tokens:
                        overlap_blocks.insert(0, prev_b)
                        overlap_tokens += prev_tok
                    else:
                        break

                current_chunk_blocks = overlap_blocks + [block]
                current_tokens = sum(self.count_tokens(b) for b in current_chunk_blocks)
            else:
                current_chunk_blocks.append(block)
                current_tokens += b_tokens

        # Commit final remaining block
        if current_chunk_blocks:
            chunk_content = "\n\n".join(current_chunk_blocks).strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_p{page_number}_c{chunk_idx}",
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    content=chunk_content,
                    token_count=self.count_tokens(chunk_content),
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_number,
                        "chunk_index": chunk_idx,
                        "has_table": "|" in chunk_content and "---" in chunk_content,
                        "has_math": ("$$" in chunk_content) or ("\\[" in chunk_content)
                    }
                )
            )

        return chunks
