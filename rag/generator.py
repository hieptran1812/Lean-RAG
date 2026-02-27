"""
Citation-grounded answer generator using OpenAI.

Precision strategy:
- System prompt forces the model to ONLY use provided context
- Each claim must cite a source chunk
- Explicit "I don't know" when context is insufficient
"""

from __future__ import annotations

import logging

from openai import OpenAI

from config import GeneratorConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document analysis assistant. Your answers must be STRICTLY grounded in the provided context passages.

RULES:
1. ONLY use information from the provided context passages to answer the question.
2. Cite your sources using [Source: filename | Section: section] format after each claim.
3. If the context does not contain enough information to fully answer the question, explicitly state what information is missing.
4. If the context contains NO relevant information, respond with: "I cannot answer this question based on the available documents."
5. For numerical data (financial figures, dates, percentages), quote the exact values from the context.
6. Do NOT infer, speculate, or add information beyond what is in the context.
7. If multiple context passages provide conflicting information, note the discrepancy.
8. Preserve the precision of financial figures — do not round or approximate unless asked.

FORMAT:
- Use clear, structured answers
- Use bullet points for multiple items
- Include citations inline after each factual claim"""


class AnswerGenerator:
    """Generates grounded, cited answers from retrieved context."""

    def __init__(self, api_key: str, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()
        self.client = OpenAI(api_key=api_key)

    def generate(self, query: str, retrieved_chunks: list[dict]) -> dict:
        """
        Generate a grounded answer.

        Returns:
            dict with keys: answer, sources, model, usage
        """
        if not retrieved_chunks:
            return {
                "answer": "I cannot answer this question — no relevant documents were found.",
                "sources": [],
                "model": self.config.model,
                "usage": {},
            }

        context = self._format_context(retrieved_chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\n---\n\nQuestion: {query}",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        answer = response.choices[0].message.content.strip()

        sources = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            sources.append(
                {
                    "source_file": meta.get("source_file", "unknown"),
                    "section": meta.get("section_hierarchy", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": chunk.get("score", chunk.get("rrf_score", 0)),
                    "text": chunk.get("text", ""),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "context_used": [c["text"][:200] + "..." for c in retrieved_chunks],
            "model": self.config.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def generate_with_history(
        self,
        query: str,
        retrieved_chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> dict:
        """
        Generate a grounded answer with conversation history for follow-ups.

        Args:
            query: Current user question.
            retrieved_chunks: Retrieved context chunks.
            chat_history: List of {"role": "user"|"assistant", "content": str} dicts.

        Returns:
            dict with keys: answer, sources, model, usage
        """
        if not retrieved_chunks:
            return {
                "answer": "I cannot answer this question — no relevant documents were found.",
                "sources": [],
                "model": self.config.model,
                "usage": {},
            }

        context = self._format_context(retrieved_chunks)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject prior conversation turns so the model can resolve references
        if chat_history:
            messages.extend(chat_history)

        messages.append(
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\n---\n\nQuestion: {query}",
            }
        )

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        answer = response.choices[0].message.content.strip()

        sources = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            sources.append(
                {
                    "source_file": meta.get("source_file", "unknown"),
                    "section": meta.get("section_hierarchy", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": chunk.get("score", chunk.get("rrf_score", 0)),
                    "text": chunk.get("text", ""),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "context_used": [c["text"][:200] + "..." for c in retrieved_chunks],
            "model": self.config.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source_file", "unknown")
            section = meta.get("section_hierarchy", "")
            text = chunk["text"]

            header = f"[Passage {i}] Source: {source}"
            if section:
                header += f" | Section: {section}"

            parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(parts)
