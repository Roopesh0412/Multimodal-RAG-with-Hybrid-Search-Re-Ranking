"""
Production Latency Benchmark & SLA Validation Script.
Validates the <400ms retrieval SLA target across Dense, Sparse BM25, RRF, and Cross-Encoder Re-Ranking.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Configure UTF-8 encoding for cross-platform console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rag_service import RAGService
from sample_docs.generate_sample import create_sample_scientific_pdf


def run_latency_benchmark(num_iterations: int = 10):
    print("=" * 80)
    print("[*] MULTIMODAL ENTERPRISE RAG: RETRIEVAL LATENCY BENCHMARK & SLA PROFILER")
    print("=" * 80)

    # 1. Prepare sample PDF
    pdf_path = PROJECT_ROOT / "sample_docs" / "enterprise_benchmark_report.pdf"
    if not pdf_path.exists():
        print(f"Creating sample scientific benchmark PDF at {pdf_path}...")
        create_sample_scientific_pdf(pdf_path)

    # 2. Initialize RAG Service
    print("Initializing RAG Service...")
    service = RAGService()

    # 3. Index Document
    print("Indexing benchmark document into ChromaDB & BM25...")
    ext_result, chunks, index_duration_ms = service.index_document(pdf_path)
    print(f"[OK] Ingested {len(chunks)} chunks, {ext_result.total_tables} tables in {index_duration_ms:.1f}ms\n")

    # Benchmark test queries
    benchmark_queries = [
        "What is the time-dependent Schrodinger equation and Planck constant in Table 1?",
        "What is the mean latency of Dense Vector Search in Table 2?",
        "What is the Dirac Hamiltonian equation for spin-1/2 fermionic fields?",
        "What is the fine-structure constant alpha uncertainty in ppm?",
        "What is the end-to-end retrieval SLA ceiling target in Table 2?"
    ]

    # Warmup round
    print("Executing warmup query...")
    _ = service.query(benchmark_queries[0], top_k=5)

    dense_times: List[float] = []
    sparse_times: List[float] = []
    rrf_times: List[float] = []
    rerank_times: List[float] = []
    total_retrieval_times: List[float] = []

    print(f"Running {num_iterations} benchmark profiling cycles across {len(benchmark_queries)} query archetypes...")

    for i in range(num_iterations):
        for q in benchmark_queries:
            # Measure Dense
            t0 = time.perf_counter()
            dense_res = service.retriever.search_dense(q, top_k=service.settings.DENSE_TOP_K)
            t_dense = (time.perf_counter() - t0) * 1000
            dense_times.append(t_dense)

            # Measure Sparse
            t0 = time.perf_counter()
            sparse_res = service.retriever.search_sparse(q, top_k=service.settings.SPARSE_TOP_K)
            t_sparse = (time.perf_counter() - t0) * 1000
            sparse_times.append(t_sparse)

            # Measure RRF
            t0 = time.perf_counter()
            candidates = service.retriever.retrieve(q, top_candidates_limit=25)
            t_rrf_total = (time.perf_counter() - t0) * 1000
            t_rrf = max(0.1, t_rrf_total - t_dense - t_sparse)
            rrf_times.append(t_rrf)

            # Measure Cross-Encoder Re-rank
            t0 = time.perf_counter()
            reranked = service.reranker.rerank(q, candidates, top_k=5)
            t_rerank = (time.perf_counter() - t0) * 1000
            rerank_times.append(t_rerank)

            total_retrieval = t_rrf_total + t_rerank
            total_retrieval_times.append(total_retrieval)

    # Metrics Summary
    print("\n" + "=" * 80)
    print("[+] BENCHMARK RESULTS (LATENCY SLA PROFILING)")
    print("=" * 80)
    print(f"{'Pipeline Stage':<28} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Mean (ms)':<10} | {'SLA Target':<12} | {'Status'}")
    print("-" * 80)

    stages = [
        ("Dense Vector Search", dense_times, 100.0),
        ("Sparse BM25 Search", sparse_times, 50.0),
        ("Reciprocal Rank Fusion", rrf_times, 10.0),
        ("Cross-Encoder Re-rank", rerank_times, 200.0),
        ("Total Retrieval + Rerank", total_retrieval_times, 400.0),
    ]

    all_passed = True
    for name, times, sla_limit in stages:
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        mean = np.mean(times)
        status = "PASS" if p95 < sla_limit else "FAIL"
        if p95 >= sla_limit:
            all_passed = False
        print(f"{name:<28} | {p50:<10.2f} | {p95:<10.2f} | {mean:<10.2f} | {f'< {sla_limit:.0f} ms':<12} | {status}")

    print("-" * 80)
    if all_passed:
        print("[SLA VERIFIED]: ALL PIPELINE STAGES SATISFIED PRODUCTION SLA (<400ms RETRIEVAL TARGET)!")
    else:
        print("[WARNING]: One or more stages exceeded SLA thresholds.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_latency_benchmark(num_iterations=3)
