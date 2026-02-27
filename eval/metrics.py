"""
Evaluation metrics for RAG system quality.

Measures:
- Retrieval quality: Context Precision, Context Recall, Hit Rate, MRR
- Answer quality: Faithfulness, Answer Relevance, Correctness
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI, OpenAI

from config import EvalConfig

logger = logging.getLogger(__name__)


# ── Retrieval Metrics (deterministic) ──────────────────────────────────


def context_precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """
    Precision@K: fraction of top-k retrieved chunks that are relevant.
    """
    top_k = retrieved_chunk_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_chunk_ids)
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / len(top_k)


def context_recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """
    Recall@K: fraction of relevant chunks that appear in top-k.
    """
    if not relevant_chunk_ids:
        return 1.0  # No relevant docs = trivially satisfied
    top_k = set(retrieved_chunk_ids[:k])
    hits = sum(1 for cid in relevant_chunk_ids if cid in top_k)
    return hits / len(relevant_chunk_ids)


def hit_rate(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """
    Hit Rate@K: 1 if any relevant chunk is in top-k, 0 otherwise.
    """
    top_k = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)
    return 1.0 if top_k & relevant_set else 0.0


def mean_reciprocal_rank(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> float:
    """
    MRR: 1/rank of the first relevant chunk.
    """
    relevant_set = set(relevant_chunk_ids)
    for i, cid in enumerate(retrieved_chunk_ids, 1):
        if cid in relevant_set:
            return 1.0 / i
    return 0.0


# ── Answer Quality Metrics (LLM-judged) ───────────────────────────────


class LLMJudge:
    """Uses an LLM to evaluate answer quality (sync and async)."""

    def __init__(self, api_key: str, config: EvalConfig | None = None):
        self.config = config or EvalConfig()
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

    def faithfulness(self, answer: str, context: str) -> dict:
        """
        Faithfulness: Is every claim in the answer supported by the context?
        Returns score 0-1 and explanation.
        """
        prompt = f"""Evaluate the faithfulness of the following answer with respect to the given context.

Faithfulness measures whether EVERY claim in the answer is supported by the provided context.

Context:
{context}

Answer:
{answer}

Scoring:
- 1.0: Every claim is directly supported by the context
- 0.75: Most claims are supported, minor unsupported details
- 0.5: Some claims are supported, some are not
- 0.25: Few claims are supported
- 0.0: The answer contradicts or is unrelated to the context

Respond with JSON: {{"score": <float>, "explanation": "<brief explanation>", "unsupported_claims": ["<claim1>", ...]}}"""

        return self._judge(prompt)

    def answer_relevance(self, answer: str, question: str) -> dict:
        """
        Answer Relevance: Does the answer address the question?
        """
        prompt = f"""Evaluate how relevant the answer is to the given question.

Question: {question}

Answer: {answer}

Scoring:
- 1.0: Directly and completely answers the question
- 0.75: Mostly answers the question with minor gaps
- 0.5: Partially answers the question
- 0.25: Tangentially related but doesn't answer the question
- 0.0: Completely irrelevant or "I don't know" response

Respond with JSON: {{"score": <float>, "explanation": "<brief explanation>"}}"""

        return self._judge(prompt)

    def correctness(self, answer: str, ground_truth: str, question: str) -> dict:
        """
        Correctness: Does the answer match the ground truth?
        """
        prompt = f"""Evaluate the correctness of the generated answer compared to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Scoring:
- 1.0: Semantically equivalent, all key facts match
- 0.75: Mostly correct, minor factual differences
- 0.5: Partially correct, some key facts missing or wrong
- 0.25: Mostly incorrect but has some relevant information
- 0.0: Completely incorrect or contradicts ground truth

For numerical answers, exact values must match (no rounding).

Respond with JSON: {{"score": <float>, "explanation": "<brief explanation>", "key_differences": ["<diff1>", ...]}}"""

        return self._judge(prompt)

    def _judge(self, prompt: str) -> dict:
        """Run a judgment prompt and parse the response."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"score": 0, "explanation": f"Could not parse: {content[:100]}"}

        except Exception as e:
            logger.warning(f"LLM judgment failed: {e}")
            return {"score": 0, "explanation": f"Error: {str(e)}"}

    # ── Async variants for concurrent evaluation ───────────────────────

    async def _ajudge(self, prompt: str) -> dict:
        """Async version of _judge."""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"score": 0, "explanation": f"Could not parse: {content[:100]}"}

        except Exception as e:
            logger.warning(f"LLM judgment failed: {e}")
            return {"score": 0, "explanation": f"Error: {str(e)}"}

    async def afaithfulness(self, answer: str, context: str) -> dict:
        """Async faithfulness evaluation."""
        prompt = f"""Evaluate the faithfulness of the following answer with respect to the given context.

Faithfulness measures whether EVERY claim in the answer is supported by the provided context.

Context:
{context}

Answer:
{answer}

Scoring:
- 1.0: Every claim is directly supported by the context
- 0.75: Most claims are supported, minor unsupported details
- 0.5: Some claims are supported, some are not
- 0.25: Few claims are supported
- 0.0: The answer contradicts or is unrelated to the context

Respond with JSON: {{"score": <float>, "explanation": "<brief explanation>", "unsupported_claims": ["<claim1>", ...]}}"""
        return await self._ajudge(prompt)

    async def aanswer_relevance(self, answer: str, question: str) -> dict:
        """Async answer relevance evaluation."""
        prompt = f"""Evaluate how relevant the answer is to the given question.

Question: {question}

Answer: {answer}

Scoring:
- 1.0: Directly and completely answers the question
- 0.75: Mostly answers the question with minor gaps
- 0.5: Partially answers the question
- 0.25: Tangentially related but doesn't answer the question
- 0.0: Completely irrelevant or "I don't know" response

Respond with JSON: {{"score": <float>, "explanation": "<brief explanation>"}}"""
        return await self._ajudge(prompt)

    async def acorrectness(self, answer: str, ground_truth: str, question: str) -> dict:
        """Async correctness evaluation."""
        prompt = f"""Evaluate the correctness of the generated answer compared to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Scoring:
- 1.0: Semantically equivalent, all key facts match
- 0.75: Mostly correct, minor factual differences
- 0.5: Partially correct, some key facts missing or wrong
- 0.25: Mostly incorrect but has some relevant information
- 0.0: Completely incorrect or contradicts ground truth

For numerical answers, exact values must match (no rounding).

Respond with JSON: {{"score": <float>, "explanation": "<brief explanation>", "key_differences": ["<diff1>", ...]}}"""
        return await self._ajudge(prompt)
