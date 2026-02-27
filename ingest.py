"""
Document ingestion entry point.
Converts markdown documents into embeddings and stores them in the vector store.

Usage:
    python ingest.py              # Ingest from default markdown_output/
    python ingest.py --reset      # Clear store and re-ingest
    python ingest.py --dir /path  # Ingest from custom directory
"""

import argparse
import logging
import sys
from pathlib import Path

from config import get_config
from rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--dir", type=str, help="Source directory with .md files")
    parser.add_argument("--reset", action="store_true", help="Clear existing chunks before ingesting")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    pipeline = RAGPipeline(config)

    source_dir = Path(args.dir) if args.dir else None

    count = pipeline.ingest(source_dir=source_dir, reset=args.reset)
    print(f"\nTotal chunks in vector store: {pipeline.vector_store.count}")

    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
