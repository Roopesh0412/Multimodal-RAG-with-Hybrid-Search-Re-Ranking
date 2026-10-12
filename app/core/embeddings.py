"""
Embedding Services for Multimodal Enterprise RAG.
Supports local HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
with ONNXRuntime acceleration and an optional toggle for OpenAI (text-embedding-3-small).
"""

import logging
from abc import ABC, abstractmethod
from typing import List
import numpy as np

logger = logging.getLogger("rag_enterprise.embeddings")


class BaseEmbeddingService(ABC):
    """Abstract base class for text embedding generation."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text documents/chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimension."""
        pass


class ONNXMiniLMEmbedder:
    """
    Native ONNXRuntime inference engine for sentence-transformers/all-MiniLM-L6-v2.
    Runs with high throughput, zero PyTorch DLL dependency, and 384-dimensional normalized vectors.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._session = None
        self._tokenizer = None

    def _load(self):
        if self._session is None:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = hf_hub_download(repo_id=self.model_name, filename="onnx/model.onnx")
            tokenizer_path = hf_hub_download(repo_id=self.model_name, filename="tokenizer.json")

            # Configure high-performance ONNX session
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_padding(length=256)
            self._tokenizer.enable_truncation(max_length=256)
            logger.info(f"Loaded ONNXRuntime engine for '{self.model_name}' (384 dimensions).")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        self._load()

        all_embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = [self._tokenizer.encode(t) for t in batch]

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

            outputs = self._session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            })

            token_embeddings = outputs[0]  # shape: (batch_size, seq_len, 384)
            mask_expanded = np.expand_dims(attention_mask, -1).repeat(token_embeddings.shape[-1], -1)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask

            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            all_embeddings.append(normalized)

        return np.vstack(all_embeddings)


class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """
    Local sentence-transformers embedding provider.
    Tries native SentenceTransformer and falls back seamlessly to ONNXRuntime acceleration.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        self._onnx_model = None
        self._dim = 384
        logger.info(f"Initialized HuggingFace embedding engine with model: {model_name}")

    def _load_model(self):
        if self._st_model is not None or self._onnx_model is not None:
            return

        # Attempt standard SentenceTransformer
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            self._dim = self._st_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded standard SentenceTransformer '{self.model_name}' (dim={self._dim})")
            return
        except Exception as e:
            logger.info(f"SentenceTransformer standard load bypassed ({e}). Utilizing ONNXRuntime acceleration.")

        # Fallback to ONNXRuntime
        try:
            self._onnx_model = ONNXMiniLMEmbedder(self.model_name)
            self._dim = 384
            logger.info(f"Active embedding provider: ONNXRuntime MiniLM engine.")
        except Exception as e2:
            logger.error(f"Failed to initialize embedding model: {e2}")
            raise RuntimeError(f"Could not load embedding engine for {self.model_name}: {e2}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()
        if self._st_model is not None:
            embeddings = self._st_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        else:
            return self._onnx_model.encode(texts).tolist()

    def embed_query(self, query: str) -> List[float]:
        self._load_model()
        if self._st_model is not None:
            embedding = self._st_model.encode(
                query,
                normalize_embeddings=True
            )
            return embedding.tolist()
        else:
            result = self._onnx_model.encode([query])
            return result[0].tolist()


class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    Cloud OpenAI embedding provider.
    Default: 'text-embedding-3-small' (1536 dimensions).
    """

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small", base_url: str = None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self._client = None
        self._dim = 1536 if "large" not in model_name else 3072

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        return self._client

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            input=[query],
            model=self.model_name
        )
        return response.data[0].embedding


def get_embedding_service(
    provider: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    api_key: str = "",
    base_url: str = None
) -> BaseEmbeddingService:
    """Factory creating configured embedding service."""
    if provider.lower() == "openai":
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. Falling back to local HuggingFace embeddings.")
            return HuggingFaceEmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return OpenAIEmbeddingService(api_key=api_key, model_name=model_name, base_url=base_url)
    return HuggingFaceEmbeddingService(model_name=model_name)
