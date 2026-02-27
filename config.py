"""
Central configuration for the RAG system.
All tunable parameters in one place for easy maintenance.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PathConfig:
    """File system paths."""

    root: Path = Path(__file__).parent
    documents_dir: Path = field(default_factory=lambda: Path(__file__).parent / "documents")
    markdown_dir: Path = field(default_factory=lambda: Path(__file__).parent / "markdown_output")
    vector_store_dir: Path = field(default_factory=lambda: Path(__file__).parent / "vector_store")
    eval_dir: Path = field(default_factory=lambda: Path(__file__).parent / "eval_data")


@dataclass(frozen=True)
class ChunkConfig:
    """Document chunking parameters."""

    chunk_size: int = 1024  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between adjacent chunks
    min_chunk_size: int = 100  # Minimum chunk size to keep
    separators: tuple = ("\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ")


@dataclass(frozen=True)
class EmbeddingConfig:
    """OpenAI embedding parameters."""

    model: str = "text-embedding-3-large"
    dimensions: int = 1536  # Reduced from 3072 for cost/speed balance
    batch_size: int = 100


@dataclass(frozen=True)
class RetrieverConfig:
    """Retrieval parameters."""

    semantic_top_k: int = 20  # Initial semantic retrieval count
    bm25_top_k: int = 20  # Initial BM25 retrieval count
    rerank_top_k: int = 5  # Final top-k after reranking
    semantic_weight: float = 0.5  # Weight for semantic score in hybrid fusion
    bm25_weight: float = 0.5  # Weight for BM25 score in hybrid fusion
    similarity_threshold: float = 0.3  # Minimum similarity to include


@dataclass(frozen=True)
class GeneratorConfig:
    """OpenAI generation parameters."""

    model: str = "gpt-4o"
    temperature: float = 0.1  # Low temperature for factual precision
    max_tokens: int = 2048
    rerank_model: str = "gpt-4o-mini"  # Cheaper model for reranking


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation parameters."""

    num_questions_per_doc: int = 10
    eval_model: str = "gpt-4o"
    eval_batch_size: int = 5


@dataclass(frozen=True)
class RAGConfig:
    """Top-level config aggregating all sub-configs."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    paths: PathConfig = field(default_factory=PathConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")


def get_config() -> RAGConfig:
    """Factory function for the global config."""
    return RAGConfig()
