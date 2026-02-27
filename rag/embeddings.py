"""
OpenAI embeddings wrapper with batching and caching.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings using OpenAI's API with batching support."""

    def __init__(self, api_key: str, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with automatic batching."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.config.model,
                dimensions=self.config.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Embedded batch {i // self.config.batch_size + 1} ({len(batch)} texts)")

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.config.model,
            dimensions=self.config.dimensions,
        )
        return response.data[0].embedding
