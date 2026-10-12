"""
Utility script to generate complex sample scientific & tabular PDFs for benchmarking.
Includes mathematical formulations, structured data tables, and scientific text.
"""

from pathlib import Path
import fitz  # PyMuPDF


def create_sample_scientific_pdf(output_path: Path) -> Path:
    """Generates a multi-page PDF containing scientific text, tables, and equations."""
    doc = fitz.open()

    # ---------------- PAGE 1: Quantum Dynamics & Equations ----------------
    page1 = doc.new_page(width=595, height=842)  # A4 size

    title = "High-Performance Multimodal Enterprise RAG Benchmark Document"
    page1.insert_text((50, 60), title, fontsize=16, fontname="helv", color=(0.1, 0.2, 0.5))

    section1 = (
        "1. Theoretical Framework & Quantum Formulations\n\n"
        "In quantum mechanics, the time-dependent Schrödinger equation describes the wave function evolution:\n"
        "$$i\\hbar \\frac{\\partial}{\\partial t} \\Psi(\\mathbf{r}, t) = \\left[ -\\frac{\\hbar^2}{2m} \\nabla^2 + V(\\mathbf{r}, t) \\right] \\Psi(\\mathbf{r}, t)$$\n\n"
        "where $\\hbar$ is the reduced Planck constant, $m$ represents particle mass, and $V(\\mathbf{r}, t)$ denotes the potential field.\n\n"
        "For relativistic systems, the Dirac Hamiltonian equation governs spin-1/2 fermionic fields:\n"
        "$$H_D = c \\boldsymbol{\\alpha} \\cdot \\mathbf{p} + \\beta m c^2$$\n\n"
        "The eigenvalues of the Hamiltonian correspond to the observable stationary energy states $E_n = n \\hbar \\omega$."
    )
    page1.insert_textbox(fitz.Rect(50, 90, 545, 300), section1, fontsize=10, fontname="helv")

    # Add Table 1 on Page 1: Physical Constants & Parameters
    table_header = "Table 1: Fundamental Physical Constants and Convergence Limits"
    page1.insert_text((50, 320), table_header, fontsize=11, fontname="helv", color=(0.1, 0.3, 0.1))

    # Draw table bounding boxes and text
    rows = [
        ["Constant / Symbol", "Value", "Unit", "Uncertainty (ppm)"],
        ["Planck Constant (h)", "6.62607015e-34", "Joule-second (J*s)", "Exact (SI standard)"],
        ["Speed of Light (c)", "299792458", "Meters per second (m/s)", "Exact (defined)"],
        ["Electron Mass (m_e)", "9.1093837e-31", "Kilograms (kg)", "0.003 ppm"],
        ["Boltzmann Constant (k_B)", "1.380649e-23", "Joules per Kelvin (J/K)", "Exact (SI standard)"],
        ["Fine-Structure Const (alpha)", "0.007297352569", "Dimensionless", "0.00015 ppm"]
    ]

    y_start = 340
    col_x = [50, 200, 340, 470, 545]
    row_h = 24

    for r_idx, row in enumerate(rows):
        y_curr = y_start + r_idx * row_h
        # Background for header
        if r_idx == 0:
            page1.draw_rect(fitz.Rect(50, y_curr, 545, y_curr + row_h), color=(0.8, 0.85, 0.95), fill=(0.88, 0.92, 0.98))
        else:
            page1.draw_rect(fitz.Rect(50, y_curr, 545, y_curr + row_h), color=(0.8, 0.8, 0.8))

        for c_idx, cell in enumerate(row):
            rect = fitz.Rect(col_x[c_idx] + 4, y_curr + 4, col_x[c_idx + 1] - 4, y_curr + row_h - 4)
            page1.insert_textbox(rect, cell, fontsize=8, fontname="helv")

    # Discussion following table
    disc = (
        "Statistical convergence was verified across 10,000 Monte Carlo trials. "
        "Notice that fine-structure constant alpha demonstrates stability within 0.00015 ppm."
    )
    page1.insert_textbox(fitz.Rect(50, 510, 545, 600), disc, fontsize=10, fontname="helv")

    # ---------------- PAGE 2: Financial & Computational Benchmark ----------------
    page2 = doc.new_page(width=595, height=842)

    title2 = "2. Financial Risk Metrics & Architectural Benchmarks"
    page2.insert_text((50, 60), title2, fontsize=14, fontname="helv", color=(0.1, 0.2, 0.5))

    desc2 = (
        "Quantitative options pricing utilizes the Black-Scholes partial differential equation:\n"
        "$$\\frac{\\partial V}{\\partial t} + \\frac{1}{2} \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} + r S \\frac{\\partial V}{\\partial S} - r V = 0$$\n\n"
        "Here, $S$ denotes asset spot price, $\\sigma$ represents annualized volatility, and $r$ is the risk-free interest rate."
    )
    page2.insert_textbox(fitz.Rect(50, 90, 545, 230), desc2, fontsize=10, fontname="helv")

    # Add Table 2: Benchmark Latency Results
    table_header2 = "Table 2: Enterprise Hybrid RAG Performance Latency SLA Benchmarks"
    page2.insert_text((50, 240), table_header2, fontsize=11, fontname="helv", color=(0.1, 0.3, 0.1))

    rows2 = [
        ["Pipeline Stage", "Method / Engine", "Mean Latency (ms)", "SLA Target (ms)", "Pass / Fail"],
        ["Dense Vector Search", "ChromaDB HNSW Cosine", "42.5 ms", "< 100 ms", "PASSED"],
        ["Sparse Lexical Search", "BM25Okapi Tokenized", "18.3 ms", "< 50 ms", "PASSED"],
        ["Reciprocal Rank Fusion", "RRF (k = 60)", "2.1 ms", "< 10 ms", "PASSED"],
        ["Cross-Encoder Rerank", "ms-marco-MiniLM-L-6-v2", "115.4 ms", "< 200 ms", "PASSED"],
        ["End-to-End Retrieval", "Hybrid + Rerank Total", "178.3 ms", "< 400 ms", "PASSED"]
    ]

    y_start2 = 260
    col_x2 = [50, 190, 330, 440, 510, 545]

    for r_idx, row in enumerate(rows2):
        y_curr = y_start2 + r_idx * row_h
        if r_idx == 0:
            page2.draw_rect(fitz.Rect(50, y_curr, 545, y_curr + row_h), color=(0.8, 0.85, 0.95), fill=(0.88, 0.92, 0.98))
        else:
            page2.draw_rect(fitz.Rect(50, y_curr, 545, y_curr + row_h), color=(0.8, 0.8, 0.8))

        for c_idx, cell in enumerate(row):
            rect = fitz.Rect(col_x2[c_idx] + 3, y_curr + 4, col_x2[c_idx + 1] - 3, y_curr + row_h - 4)
            page2.insert_textbox(rect, cell, fontsize=8, fontname="helv")

    summary2 = (
        "Conclusion: The end-to-end hybrid retrieval and cross-encoder re-ranking stage operates at 178.3 ms, "
        "well within the strict production SLA ceiling of <400 ms."
    )
    page2.insert_textbox(fitz.Rect(50, 440, 545, 520), summary2, fontsize=10, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    out = Path("./sample_docs/enterprise_benchmark_report.pdf")
    create_sample_scientific_pdf(out)
    print(f"Generated sample PDF at: {out.resolve()}")
