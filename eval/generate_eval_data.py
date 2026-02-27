"""
Synthetic evaluation dataset generator.

Uses an LLM to read document chunks and generate high-quality Q&A pairs
with ground truth answers and source references for evaluating the RAG pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI, OpenAI

from config import EvalConfig, RAGConfig, get_config
from rag.chunker import Chunk, MarkdownChunker

logger = logging.getLogger(__name__)

# Max concurrent LLM calls to avoid rate limits
MAX_CONCURRENT_LLM = 5

QUESTION_GEN_PROMPT = """You are an expert at creating evaluation questions for a document Q&A system.

Given the following text passages from a document, generate {num_questions} diverse question-answer pairs that can be answered using ONLY the provided text.

REQUIREMENTS:
1. Questions should be specific and factual — not vague or open-ended.
2. Include a mix of question types:
   - Factual recall (e.g., "What is the total revenue for 2024?")
   - Comparison (e.g., "How did net income change from 2023 to 2024?")
   - Detail extraction (e.g., "When was Neksai Inc. incorporated?")
   - Relationship (e.g., "What is the ownership structure of PTE?")
3. Each answer MUST be directly supported by the text.
4. Include the exact text passage that supports the answer.
5. Rate each question's difficulty: easy, medium, hard.

Source document: {source_file}

Text passages:
{passages}

Respond with a JSON array of objects, each with these keys:
- "question": the question string
- "answer": the ground truth answer (concise but complete)
- "supporting_text": the exact quote from the passage that supports the answer
- "difficulty": "easy" | "medium" | "hard"
- "question_type": "factual" | "comparison" | "detail" | "relationship" | "numerical"
- "source_chunk_ids": list of chunk_ids used

Return ONLY valid JSON. No explanations."""


class EvalDataGenerator:
    """Generates synthetic Q&A evaluation datasets from document chunks."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or get_config()
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.config.openai_api_key)
        self.chunker = MarkdownChunker(self.config.chunk)

    # ── Sync mode ────────────────────────────────────────────────────

    def generate_eval_dataset(
        self,
        source_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> list[dict]:
        """
        Generate evaluation Q&A pairs sequentially (sync mode).

        Returns:
            List of Q&A evaluation items.
        """
        source_dir = source_dir or self.config.paths.markdown_dir
        output_path = output_path or (self.config.paths.eval_dir / "eval_dataset.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Chunking documents from {source_dir}...")
        all_chunks = self.chunker.chunk_directory(source_dir)

        # Group chunks by source file
        chunks_by_source: dict[str, list[Chunk]] = {}
        for chunk in all_chunks:
            chunks_by_source.setdefault(chunk.source_file, []).append(chunk)

        all_qa_pairs: list[dict] = []

        for source_file, chunks in chunks_by_source.items():
            print(f"\nGenerating questions for: {source_file}")
            qa_pairs = self._generate_for_document(source_file, chunks)
            all_qa_pairs.extend(qa_pairs)
            print(f"  Generated {len(qa_pairs)} Q&A pairs")

        # Save dataset
        output_path.write_text(json.dumps(all_qa_pairs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved {len(all_qa_pairs)} Q&A pairs to {output_path}")

        return all_qa_pairs

    def _generate_for_document(
        self,
        source_file: str,
        chunks: list[Chunk],
    ) -> list[dict]:
        """Generate Q&A pairs for one document's chunks (sync)."""
        qa_pairs: list[dict] = []

        window_size = 5
        step = 3
        num_questions = self.config.eval.num_questions_per_doc

        num_windows = max(1, (len(chunks) - window_size) // step + 1)
        questions_per_window = max(2, num_questions // num_windows)

        for i in range(0, len(chunks), step):
            window = chunks[i : i + window_size]
            if not window:
                break

            passages = self._format_passages(window)

            prompt = QUESTION_GEN_PROMPT.format(
                num_questions=questions_per_window,
                source_file=source_file,
                passages=passages,
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.config.eval.eval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4096,
                )

                content = response.choices[0].message.content.strip()
                parsed = self._parse_json_response(content)

                for item in parsed:
                    item["source_file"] = source_file
                    item["source_chunk_ids"] = [c.chunk_id for c in window]
                    qa_pairs.append(item)

            except Exception as e:
                logger.warning(f"Failed to generate questions for window {i}: {e}")
                continue

            if len(qa_pairs) >= num_questions:
                break

        return qa_pairs[:num_questions]

    # ── Async mode ─────────────────────────────────────────────────

    def generate_eval_dataset_async(
        self,
        source_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> list[dict]:
        """Sync entry point that runs the async generation internally."""
        return asyncio.run(self._agenerate_eval_dataset(source_dir, output_path))

    async def _agenerate_eval_dataset(
        self,
        source_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> list[dict]:
        """
        Generate evaluation Q&A pairs from all markdown documents concurrently.

        Returns:
            List of Q&A evaluation items.
        """
        source_dir = source_dir or self.config.paths.markdown_dir
        output_path = output_path or (self.config.paths.eval_dir / "eval_dataset.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear old eval dataset if it exists
        if output_path.exists():
            logger.info(f"Clearing old eval dataset: {output_path}")
            print(f"Clearing old eval dataset: {output_path}")
            output_path.unlink()

        print(f"Chunking documents from {source_dir}...")
        all_chunks = self.chunker.chunk_directory(source_dir)

        # Group chunks by source file
        chunks_by_source: dict[str, list[Chunk]] = {}
        for chunk in all_chunks:
            chunks_by_source.setdefault(chunk.source_file, []).append(chunk)

        # Process all documents concurrently
        tasks = [
            self._agenerate_for_document(source_file, chunks)
            for source_file, chunks in chunks_by_source.items()
        ]
        doc_results = await asyncio.gather(*tasks)

        all_qa_pairs: list[dict] = []
        for (source_file, _), qa_pairs in zip(chunks_by_source.items(), doc_results):
            print(f"  {source_file}: {len(qa_pairs)} Q&A pairs")
            all_qa_pairs.extend(qa_pairs)

        # Save dataset
        output_path.write_text(json.dumps(all_qa_pairs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved {len(all_qa_pairs)} Q&A pairs to {output_path}")

        return all_qa_pairs

    async def _agenerate_for_document(
        self,
        source_file: str,
        chunks: list[Chunk],
    ) -> list[dict]:
        """Generate Q&A pairs for one document's chunks concurrently."""
        window_size = 5
        step = 3
        num_questions = self.config.eval.num_questions_per_doc

        num_windows = max(1, (len(chunks) - window_size) // step + 1)
        questions_per_window = max(2, num_questions // num_windows)

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
        tasks = []
        for i in range(0, len(chunks), step):
            window = chunks[i : i + window_size]
            if not window:
                break
            tasks.append(
                self._agenerate_for_window(
                    semaphore, source_file, window, questions_per_window, i
                )
            )

        window_results = await asyncio.gather(*tasks)

        qa_pairs: list[dict] = []
        for items in window_results:
            qa_pairs.extend(items)
            if len(qa_pairs) >= num_questions:
                break

        return qa_pairs[:num_questions]

    async def _agenerate_for_window(
        self,
        semaphore: asyncio.Semaphore,
        source_file: str,
        window: list[Chunk],
        questions_per_window: int,
        window_idx: int,
    ) -> list[dict]:
        """Generate Q&A pairs for a single chunk window."""
        async with semaphore:
            passages = self._format_passages(window)
            prompt = QUESTION_GEN_PROMPT.format(
                num_questions=questions_per_window,
                source_file=source_file,
                passages=passages,
            )

            try:
                response = await self.async_client.chat.completions.create(
                    model=self.config.eval.eval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4096,
                )

                content = response.choices[0].message.content.strip()
                parsed = self._parse_json_response(content)

                for item in parsed:
                    item["source_file"] = source_file
                    item["source_chunk_ids"] = [c.chunk_id for c in window]

                return parsed
            except Exception as e:
                logger.warning(f"Failed to generate questions for window {window_idx}: {e}")
                return []

    @staticmethod
    def _format_passages(chunks: list[Chunk]) -> str:
        """Format chunks into numbered passages for the prompt."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks):
            section = " > ".join(chunk.section_hierarchy) if chunk.section_hierarchy else "N/A"
            parts.append(
                f"[Chunk {i} | ID: {chunk.chunk_id} | Section: {section}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _parse_json_response(content: str) -> list[dict]:
        """Robustly parse JSON from LLM response."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from markdown code block
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding any JSON array
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from response: {content[:200]}...")
        return []
