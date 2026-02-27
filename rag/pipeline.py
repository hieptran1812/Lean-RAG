"""
Main RAG pipeline orchestrator.
Ties together ingestion, retrieval, and generation into a clean interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import RAGConfig, get_config
from rag.chunker import Chunk, MarkdownChunker
from rag.embeddings import EmbeddingService
from rag.generator import AnswerGenerator
from rag.retriever import HybridRetriever
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: ingest → retrieve → generate."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or get_config()

        self.embedding_service = EmbeddingService(
            api_key=self.config.openai_api_key,
            config=self.config.embedding,
        )
        self.vector_store = VectorStore(self.config.paths.vector_store_dir)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            api_key=self.config.openai_api_key,
            retriever_config=self.config.retriever,
            generator_config=self.config.generator,
        )
        self.generator = AnswerGenerator(
            api_key=self.config.openai_api_key,
            config=self.config.generator,
        )
        self.chunker = MarkdownChunker(self.config.chunk)

    # ── Ingestion ──────────────────────────────────────────────────────

    def ingest(self, source_dir: Path | None = None) -> int:
        """
        Ingest markdown documents into the vector store.

        Clears existing chunks before storing new ones to avoid duplicates.

        Args:
            source_dir: Directory containing .md files (defaults to config)

        Returns:
            Number of chunks ingested
        """
        source_dir = source_dir or self.config.paths.markdown_dir

        print(f"Chunking documents from {source_dir}...")
        chunks = self.chunker.chunk_directory(source_dir)

        if not chunks:
            print("No chunks produced. Check your markdown files.")
            return 0

        # Always clear old chunks before storing new ones to avoid duplicates
        self.vector_store.reset()
        logger.info("Vector store cleared before ingestion")

        print(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        print("Storing in vector store...")
        self.vector_store.add_chunks(chunks, embeddings)

        print(f"Ingestion complete: {len(chunks)} chunks stored.")
        return len(chunks)

    # ── Query ──────────────────────────────────────────────────────────

    def query(self, question: str, use_rerank: bool = True) -> dict:
        """
        Answer a question using the full RAG pipeline.

        Returns:
            dict with: answer, sources, context_used, model, usage
        """
        if self.vector_store.count == 0:
            return {
                "answer": "No documents have been ingested yet. Run ingestion first.",
                "sources": [],
            }

        # Retrieve
        retrieved = self.retriever.retrieve(question, use_rerank=use_rerank)

        # Generate
        result = self.generator.generate(question, retrieved)

        return result

    def query_with_history(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        use_rerank: bool = True,
    ) -> dict:
        """
        Answer a question while considering prior conversation turns.

        Args:
            question: Current question.
            chat_history: Previous conversation turns for follow-up context.
            use_rerank: Whether to apply LLM reranking.

        Returns:
            dict with: answer, sources, context_used, model, usage
        """
        if self.vector_store.count == 0:
            return {
                "answer": "No documents have been ingested yet. Run ingestion first.",
                "sources": [],
            }

        retrieved = self.retriever.retrieve(question, use_rerank=use_rerank)
        result = self.generator.generate_with_history(question, retrieved, chat_history)
        return result

    def retrieve_only(self, question: str, top_k: int | None = None) -> list[dict]:
        """Retrieve relevant chunks without generating an answer (useful for eval)."""
        if self.vector_store.count == 0:
            return []

        retrieved = self.retriever.retrieve(question, use_rerank=True)

        if top_k is not None:
            retrieved = retrieved[:top_k]

        return retrieved
