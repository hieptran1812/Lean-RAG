"""
ChromaDB vector store for persistent embedding storage and retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "doc_rag"


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the store."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_dict() for c in chunks]

        # ChromaDB has a batch limit; add in batches of 5000
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        logger.info(f"Added {len(chunks)} chunks to vector store")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[dict]:
        """Query the vector store and return results with scores."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                hits.append(
                    {
                        "chunk_id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i],  # cosine similarity
                    }
                )

        return hits

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """Retrieve specific documents by their chunk IDs."""
        if not ids:
            return []
        results = self.collection.get(ids=ids, include=["documents", "metadatas"])
        docs: list[dict] = []
        for i in range(len(results["ids"])):
            docs.append(
                {
                    "chunk_id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )
        return docs

    def get_all_documents(self) -> list[dict]:
        """Retrieve all documents (for BM25 index building)."""
        results = self.collection.get(include=["documents", "metadatas"])
        docs: list[dict] = []
        for i in range(len(results["ids"])):
            docs.append(
                {
                    "chunk_id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )
        return docs

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset")
