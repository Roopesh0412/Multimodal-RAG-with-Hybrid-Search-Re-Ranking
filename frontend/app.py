"""
Modern Web Interface for Multimodal Enterprise RAG.
Features drag-and-drop PDF ingestion, live indexing metrics, interactive chat,
rich table/LaTeX rendering, and detailed grounding citation cards.
"""

import os
import sys
import time
import requests
import streamlit as st
from pathlib import Path

# Add project root to sys.path so frontend can import core modules if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Configure Streamlit page
st.set_page_config(
    page_title="Enterprise Multimodal RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #f1f5f9;
        color: #334155;
        margin-right: 8px;
        border: 1px solid #e2e8f0;
    }
    .citation-card {
        padding: 12px;
        border-radius: 8px;
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    .citation-source {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 4px;
    }
    .confidence-tag {
        float: right;
        font-weight: 600;
        color: #059669;
    }
</style>
""", unsafe_allow_html=True)


def get_health():
    """Fetch health and stats from FastAPI backend or return fallback."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if resp.status_code == 200:
            return resp.json(), True
    except Exception:
        pass
    return None, False


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []


# ==============================================================================
# SIDEBAR: DOCUMENT INGESTION & CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🔬 System Dashboard")
    health_data, is_api_online = get_health()

    if is_api_online:
        st.success("🟢 FastAPI Backend: Online")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", health_data.get("total_chunks", 0))
        with col2:
            st.metric("Documents", health_data.get("documents_indexed", 0))

        st.caption(f"**Embedder:** `{health_data.get('embedding_model', 'all-MiniLM-L6-v2').split('/')[-1]}`")
        st.caption(f"**Re-ranker:** `{health_data.get('reranker_model', 'ms-marco-MiniLM-L-6-v2').split('/')[-1]}`")
        st.caption(f"**LLM:** `{health_data.get('llm_model', 'gpt-4o-mini')}`")
    else:
        st.warning("🟡 Backend Offline (Local Standalone Mode)")

    st.markdown("---")
    st.markdown("### 📄 PDF Document Ingestion")
    st.caption("Extracts tables (Markdown), LaTeX formulas, and runs OCR fallback for scanned pages.")

    uploaded_files = st.file_uploader(
        "Upload Scientific / Tabular PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process & Index Documents", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Indexing '{uploaded_file.name}'...")
            try:
                if is_api_online:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=120)
                    if resp.status_code == 201:
                        data = resp.json()
                        st.session_state.indexed_docs.append(data)
                        st.success(f"Indexed **{uploaded_file.name}**: {data['total_chunks']} chunks, {data['total_tables']} tables ({data['processing_time_ms']}ms)")
                    else:
                        st.error(f"Error indexing {uploaded_file.name}: {resp.text}")
                else:
                    st.info("Start the FastAPI backend server (`python -m uvicorn app.main:app --reload`) to index documents via API.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("Ingestion completed!")
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Search Configuration")
    top_k_slider = st.slider("Re-ranked Contexts (Top-K)", min_value=1, max_value=10, value=5)
    filter_doc = st.text_input("Filter by Document ID (Optional)", placeholder="Leave blank for all")


# ==============================================================================
# MAIN CHAT & QUERY INTERFACE
# ==============================================================================
st.markdown('<div class="main-header">Multimodal Enterprise RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Production-grade Hybrid Search (Dense + Sparse BM25 + RRF) with Cross-Encoder Re-Ranking '
    'optimized for scientific, mathematical, and tabular documents.'
    '</div>',
    unsafe_allow_html=True
)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display Latency Metrics & Citations if present
        if "metrics" in message and message["metrics"]:
            m = message["metrics"]
            st.markdown(
                f'<span class="metric-badge">⚡ Total: {m.get("total_latency_ms", 0)}ms</span>'
                f'<span class="metric-badge">🔍 Retrieval: {m.get("retrieval_latency_ms", 0)}ms</span>'
                f'<span class="metric-badge">🎯 Re-rank: {m.get("rerank_latency_ms", 0)}ms</span>'
                f'<span class="metric-badge">🤖 Gen: {m.get("generation_latency_ms", 0)}ms</span>',
                unsafe_allow_html=True
            )

        if "citations" in message and message["citations"]:
            with st.expander(f"📚 Grounding Sources & Evidence ({len(message['citations'])} citations)"):
                for cit in message["citations"]:
                    conf = f"{cit.get('relevance_score', 0) * 100:.1f}%"
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-source">
                            📄 {cit.get('filename')} (Page {cit.get('page_number')})
                            <span class="confidence-tag">Confidence: {conf}</span>
                        </div>
                        <div style="font-style: italic; color: #475569; margin-top: 6px;">
                            "{cit.get('snippet')}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# User Chat Input
if prompt := st.chat_input("Ask a question about your uploaded documents, tables, or formulas..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Executing Hybrid Retrieval & Cross-Encoder Re-Ranking..."):
            try:
                payload = {
                    "question": prompt,
                    "top_k": top_k_slider,
                    "filter_document_id": filter_doc if filter_doc.strip() else None,
                    "stream": False
                }
                resp = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])
                    metrics = data.get("metrics", {})

                    st.markdown(answer)

                    # Display latency metrics
                    st.markdown(
                        f'<span class="metric-badge">⚡ Total: {metrics.get("total_latency_ms", 0)}ms</span>'
                        f'<span class="metric-badge">🔍 Retrieval: {metrics.get("retrieval_latency_ms", 0)}ms</span>'
                        f'<span class="metric-badge">🎯 Re-rank: {metrics.get("rerank_latency_ms", 0)}ms</span>'
                        f'<span class="metric-badge">🤖 Gen: {metrics.get("generation_latency_ms", 0)}ms</span>',
                        unsafe_allow_html=True
                    )

                    # Display citations
                    if citations:
                        with st.expander(f"📚 Grounding Sources & Evidence ({len(citations)} citations)"):
                            for cit in citations:
                                conf = f"{cit.get('relevance_score', 0) * 100:.1f}%"
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-source">
                                        📄 {cit.get('filename')} (Page {cit.get('page_number')})
                                        <span class="confidence-tag">Confidence: {conf}</span>
                                    </div>
                                    <div style="font-style: italic; color: #475569; margin-top: 6px;">
                                        "{cit.get('snippet')}"
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "metrics": metrics
                    })
                else:
                    err_msg = f"API error ({resp.status_code}): {resp.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"Failed to connect to backend: {e}. Ensure the FastAPI server is running at {API_BASE_URL}."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
