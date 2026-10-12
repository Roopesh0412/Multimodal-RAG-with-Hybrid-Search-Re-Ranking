"""
ChromaDB Persistent Vector Store Manager.
Supports persistent local storage, cosine similarity indexing, and rich metadata filtering.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.chunker import DocumentChunk

logger = logging.getLogger("rag_enterprise.vector_store")


class ChromaVectorStore:
    """
    Manages local persistent vector storage using ChromaDB.
    Indexed with HNSW space = 'cosine' for standard normalized cosine similarity.
    """

    COLLECTION_NAME = "enterprise_multimodal_rag"

    def __init__(self, persist_directory: Path):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB PersistentClient at {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB collection '{self.COLLECTION_NAME}' active with {self.collection.count()} chunks.")

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> int:
        """
        Inserts document chunks and corresponding embeddings into ChromaDB.
        """
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = []

        for c in chunks:
            meta = {
                "document_id": str(c.document_id),
                "filename": str(c.filename),
                "page_number": int(c.page_number),
                "chunk_id": str(c.chunk_id),
                "chunk_index": int(c.chunk_index),
                "token_count": int(c.token_count),
                "has_table": bool(c.metadata.get("has_table", False)),
                "has_math": bool(c.metadata.get("has_math", False)),
            }
            metadatas.append(meta)

        # Batch upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB.")
        return len(chunks)

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs dense semantic search via ChromaDB cosine similarity.
        Returns candidate dicts with score, id, document, and metadata.
        """
        total_items = self.collection.count()
        if total_items == 0:
            return []

        effective_k = min(top_k, total_items)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": effective_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        candidates = []
        if results and results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                # Chroma cosine distance = 1 - cosine_similarity.
                # Cosine similarity = 1 - distance. Clamped between 0 and 1.
                cos_sim = max(0.0, min(1.0, 1.0 - distances[i]))
                candidates.append({
                    "chunk_id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "dense_score": cos_sim,
                    "distance": distances[i]
                })

        return candidates

    def get_all_chunks(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves chunks and metadata for BM25 indexing or document inspection.
        """
        where_clause = {"document_id": document_id} if document_id else None
        results = self.collection.get(
            where=where_clause,
            include=["documents", "metadatas"]
        )

        chunks_data = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                chunks_data.append({
                    "chunk_id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })

        return chunks_data

    def delete_document(self, document_id: str) -> int:
        """Deletes all chunks associated with a document_id."""
        existing = self.collection.get(where={"document_id": document_id})
        count = len(existing["ids"]) if existing and existing["ids"] else 0
        if count > 0:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted {count} chunks for document_id '{document_id}'.")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the indexed collection."""
        all_data = self.collection.get(include=["metadatas"])
        total_chunks = len(all_data["ids"]) if all_data and all_data["ids"] else 0

        unique_docs = set()
        tables_count = 0
        math_count = 0

        if all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                if meta:
                    unique_docs.add(meta.get("document_id"))
                    if meta.get("has_table"):
                        tables_count += 1
                    if meta.get("has_math"):
                        math_count += 1

        return {
            "total_chunks": total_chunks,
            "total_documents": len(unique_docs),
            "documents_indexed": list(unique_docs),
            "chunks_with_tables": tables_count,
            "chunks_with_math": math_count
        }
