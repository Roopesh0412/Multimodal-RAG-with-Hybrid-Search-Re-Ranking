"""
FastAPI Enterprise Application Entrypoint.
Configures CORS, Lifespan hooks, Structured Logging, and Middleware.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.logging_config import configure_logging
from app.api.routes import router as api_router

settings = get_settings()
logger = configure_logging(
    log_level=settings.LOG_LEVEL,
    json_logs=(settings.ENVIRONMENT == "production")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for warm-up and graceful shutdown."""
    logger.info(f"Initializing {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    logger.info(f"Storage directories: Uploads='{settings.upload_path}', ChromaDB='{settings.chroma_path}'")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
## Multimodal Enterprise RAG Engine
Production-grade Retrieval-Augmented Generation optimized for complex scientific, mathematical, and tabular PDFs.

### Features:
- **PyMuPDF Layout & Table Parser**: Automatic markdown table structuring and formula preservation.
- **OCR Fallback**: Automated fallback for scanned pages.
- **Recursive Markdown Chunker**: 800 token chunk size, 150 token overlap.
- **Hybrid Retrieval**: Dense (ChromaDB cosine) + Sparse (BM25Okapi) fused via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Re-Ranking**: 'ms-marco-MiniLM-L-6-v2' / FlashRank for high-confidence ranking.
- **OpenAI/Groq/Ollama LLM Generation**: Grounded responses with precise citations.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend applications (Streamlit, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Tracks end-to-end HTTP request latency."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# Include API Router
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """Root welcome endpoint redirecting to documentation."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "documentation": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
