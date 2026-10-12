"""
Document Ingestion & Multimodal Processing Engine.
Optimized for complex scientific, mathematical, and tabular PDFs.
Extracts native text, tables (rendered as Markdown), equations, and provides OCR fallback.
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import fitz  # PyMuPDF
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

logger = logging.getLogger("rag_enterprise.processor")


@dataclass
class ProcessedPage:
    """Represents the structured content extracted from a single document page."""
    page_number: int
    text_content: str
    tables_found: int = 0
    is_scanned: bool = False
    images_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentExtractionResult:
    """Represents the overall extraction result of a PDF document."""
    document_id: str
    filename: str
    total_pages: int
    pages: List[ProcessedPage]
    total_tables: int
    total_characters: int
    has_scanned_pages: bool


class DocumentProcessor:
    """
    High-fidelity PDF document extractor.
    Combines PyMuPDF layout analysis, table parsing to Markdown, and fallback OCR.
    """

    def __init__(
        self,
        min_text_threshold: int = 50,
        enable_ocr: bool = True
    ):
        self.min_text_threshold = min_text_threshold
        self.enable_ocr = enable_ocr
        self._paddle_ocr = None

    def _get_paddle_ocr(self):
        """Lazy load PaddleOCR model if needed."""
        if self._paddle_ocr is None and PADDLE_AVAILABLE:
            try:
                self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}. Will fallback to Tesseract.")
                self._paddle_ocr = None
        return self._paddle_ocr

    def extract_tables_as_markdown(self, page: fitz.Page) -> Tuple[List[str], List[fitz.Rect]]:
        """
        Extract tables from PyMuPDF page and format as Markdown tables.
        Returns list of markdown table strings and their bounding rectangles.
        """
        md_tables = []
        table_rects = []

        try:
            tabs = page.find_tables()
            for tab in tabs:
                df_data = tab.extract()
                if not df_data or len(df_data) < 1:
                    continue

                # Build Markdown table
                headers = [str(c).strip() if c is not None else "" for c in df_data[0]]
                if not any(headers):
                    continue

                header_row = "| " + " | ".join(headers) + " |"
                separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
                body_rows = []

                for row in df_data[1:]:
                    clean_row = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
                    body_rows.append("| " + " | ".join(clean_row) + " |")

                markdown_table = "\n".join([header_row, separator_row] + body_rows)
                md_tables.append(markdown_table)
                table_rects.append(fitz.Rect(tab.bbox))
        except Exception as e:
            logger.debug(f"Table extraction error on page {page.number}: {e}")

        return md_tables, table_rects

    def _perform_ocr(self, page: fitz.Page) -> str:
        """
        Executes OCR on a scanned page image using PaddleOCR or PyTesseract.
        """
        if not self.enable_ocr:
            return ""

        try:
            # Render page to high-res pixmap (300 DPI)
            matrix = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            # 1. Try PaddleOCR first
            paddle = self._get_paddle_ocr()
            if paddle:
                try:
                    result = paddle.ocr(img_bytes, cls=True)
                    lines = []
                    if result and len(result) > 0 and result[0]:
                        for line in result[0]:
                            text = line[1][0]
                            lines.append(text)
                    if lines:
                        return "\n".join(lines)
                except Exception as e:
                    logger.debug(f"PaddleOCR failed, attempting Tesseract: {e}")

            # 2. Fallback to PyTesseract
            if PYTESSERACT_AVAILABLE:
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    return ocr_text.strip()
                except Exception as e:
                    logger.debug(f"PyTesseract failed: {e}")

            logger.warning(f"No OCR engine succeeded for scanned page {page.number + 1}.")
            return ""
        except Exception as e:
            logger.error(f"Error executing OCR on page {page.number + 1}: {e}")
            return ""

    def process_pdf(self, file_path: Path, document_id: str) -> DocumentExtractionResult:
        """
        Processes a PDF file, extracting text, formulas, markdown tables, and OCR if required.
        """
        logger.info(f"Ingesting PDF document: {file_path.name} [ID: {document_id}]")
        doc = fitz.open(str(file_path))
        pages_result: List[ProcessedPage] = []
        total_tables = 0
        total_chars = 0
        has_scanned = False

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Extract markdown tables and their bounding boxes
            md_tables, table_rects = self.extract_tables_as_markdown(page)
            tables_count = len(md_tables)
            total_tables += tables_count

            # 2. Extract text outside detected tables to prevent duplication
            # Using get_text("blocks") allows layout preservation
            blocks = page.get_text("blocks")
            non_table_blocks = []

            for b in blocks:
                # b = (x0, y0, x1, y1, text, block_no, block_type)
                block_rect = fitz.Rect(b[0], b[1], b[2], b[3])
                # Check if block overlaps with any extracted table
                overlaps_table = any(block_rect.intersects(t_rect) for t_rect in table_rects)
                if not overlaps_table and b[4].strip():
                    non_table_blocks.append(b[4].strip())

            body_text = "\n\n".join(non_table_blocks)

            # Combine body text and extracted markdown tables
            combined_content_parts = []
            if body_text:
                combined_content_parts.append(body_text)
            if md_tables:
                combined_content_parts.append("\n\n### Extracted Tables:\n" + "\n\n".join(md_tables))

            final_text = "\n\n".join(combined_content_parts).strip()
            is_scanned_page = False

            # 3. Check if page is scanned or image-heavy
            if len(final_text) < self.min_text_threshold:
                logger.info(f"Page {page_num} has minimal text ({len(final_text)} chars). Triggering OCR.")
                ocr_text = self._perform_ocr(page)
                if ocr_text:
                    final_text = ocr_text
                    is_scanned_page = True
                    has_scanned = True

            images_count = len(page.get_images())
            total_chars += len(final_text)

            pages_result.append(
                ProcessedPage(
                    page_number=page_num,
                    text_content=final_text,
                    tables_found=tables_count,
                    is_scanned=is_scanned_page,
                    images_count=images_count,
                    metadata={
                        "page_number": page_num,
                        "document_id": document_id,
                        "source": file_path.name,
                        "char_count": len(final_text),
                        "tables_count": tables_count
                    }
                )
            )

        doc.close()
        logger.info(
            f"Completed processing '{file_path.name}': {len(pages_result)} pages, "
            f"{total_tables} tables, {total_chars} total characters."
        )

        return DocumentExtractionResult(
            document_id=document_id,
            filename=file_path.name,
            total_pages=len(pages_result),
            pages=pages_result,
            total_tables=total_tables,
            total_characters=total_chars,
            has_scanned_pages=has_scanned
        )
