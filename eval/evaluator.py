"""
Full evaluation harness for the RAG pipeline.

Runs the eval dataset through the pipeline and computes all metrics,
producing a detailed JSON report and summary statistics.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAGConfig, get_config
from eval.metrics import (
    LLMJudge,
    context_precision_at_k,
    context_recall_at_k,
    hit_rate,
    mean_reciprocal_rank,
)
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Max concurrent evaluation items (each item spawns 3 LLM judge calls)
MAX_CONCURRENT_EVAL = 3


@dataclass
class EvalResult:
    """Result for a single evaluation question."""

    question: str
    ground_truth: str
    generated_answer: str
    source_file: str
    difficulty: str
    question_type: str

    # Retrieval metrics
    context_precision: float = 0.0
    context_recall: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0

    # Answer quality metrics
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    correctness: float = 0.0

    # Details
    faithfulness_detail: dict = field(default_factory=dict)
    relevance_detail: dict = field(default_factory=dict)
    correctness_detail: dict = field(default_factory=dict)

    sources_retrieved: list[str] = field(default_factory=list)
    retrieved_chunk_texts: list[str] = field(default_factory=list)
    actual_chunk_ids: list[str] = field(default_factory=list)
    actual_chunk_texts: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "generated_answer": self.generated_answer,
            "source_file": self.source_file,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "retrieval_metrics": {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "hit_rate": self.hit_rate,
                "mrr": self.mrr,
            },
            "answer_metrics": {
                "faithfulness": self.faithfulness,
                "answer_relevance": self.answer_relevance,
                "correctness": self.correctness,
            },
            "details": {
                "faithfulness": self.faithfulness_detail,
                "relevance": self.relevance_detail,
                "correctness": self.correctness_detail,
            },
            "sources_retrieved": [
                {"chunk_id": cid, "text": txt}
                for cid, txt in zip(self.sources_retrieved, self.retrieved_chunk_texts)
            ],
            "actual_chunks": [
                {"chunk_id": cid, "text": txt}
                for cid, txt in zip(self.actual_chunk_ids, self.actual_chunk_texts)
            ],
            "latency_seconds": self.latency_seconds,
        }

    @staticmethod
    def csv_header() -> list[str]:
        return [
            "question",
            "ground_truth",
            "generated_answer",
            "source_file",
            "difficulty",
            "question_type",
            "context_precision",
            "context_recall",
            "hit_rate",
            "mrr",
            "faithfulness",
            "faithfulness_explanation",
            "answer_relevance",
            "relevance_explanation",
            "correctness",
            "correctness_explanation",
            "retrieved_chunk_ids",
            "retrieved_chunk_texts",
            "actual_chunk_ids",
            "actual_chunk_texts",
            "latency_seconds",
        ]

    def to_csv_row(self) -> list[str]:
        """Flatten EvalResult into a list of strings for CSV export."""
        sep = "\n---\n"
        return [
            self.question,
            self.ground_truth,
            self.generated_answer,
            self.source_file,
            self.difficulty,
            self.question_type,
            f"{self.context_precision:.4f}",
            f"{self.context_recall:.4f}",
            f"{self.hit_rate:.4f}",
            f"{self.mrr:.4f}",
            f"{self.faithfulness:.4f}",
            self.faithfulness_detail.get("explanation", ""),
            f"{self.answer_relevance:.4f}",
            self.relevance_detail.get("explanation", ""),
            f"{self.correctness:.4f}",
            self.correctness_detail.get("explanation", ""),
            sep.join(self.sources_retrieved),
            sep.join(self.retrieved_chunk_texts),
            sep.join(self.actual_chunk_ids),
            sep.join(self.actual_chunk_texts),
            f"{self.latency_seconds:.3f}",
        ]


class RAGEvaluator:
    """Evaluates a RAG pipeline against a ground truth dataset."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or get_config()
        self.pipeline = RAGPipeline(self.config)
        self.judge = LLMJudge(
            api_key=self.config.openai_api_key,
            config=self.config.eval,
        )

    # ── Sync mode ────────────────────────────────────────────────────

    def evaluate(
        self,
        eval_dataset_path: Path | None = None,
        output_path: Path | None = None,
    ) -> dict:
        """
        Run full evaluation sequentially (sync mode).

        Returns:
            Summary dict with aggregate metrics.
        """
        eval_dataset_path = eval_dataset_path or (self.config.paths.eval_dir / "eval_dataset.json")
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.paths.eval_dir / f"eval_report_{ts}.json"

        if not eval_dataset_path.exists():
            raise FileNotFoundError(
                f"Eval dataset not found at {eval_dataset_path}. "
                "Run `python run_eval.py --generate` first."
            )

        dataset = json.loads(eval_dataset_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(dataset)} evaluation questions")

        results: list[EvalResult] = []

        for i, item in enumerate(dataset):
            print(f"\n[{i + 1}/{len(dataset)}] {item['question'][:80]}...")

            result = self._evaluate_single(item)
            results.append(result)

            self._log_eval_result(result)

        summary = self._build_and_save_report(results, output_path)
        return summary

    def _evaluate_single(self, item: dict) -> EvalResult:
        """Evaluate a single Q&A pair (sync)."""
        question = item["question"]
        ground_truth = item["answer"]
        relevant_ids = item.get("source_chunk_ids", [])

        # Time the query
        start = time.time()
        response = self.pipeline.query(question, use_rerank=True)
        latency = time.time() - start

        answer = response.get("answer", "")
        retrieved = self.pipeline.retrieve_only(question)
        retrieved_ids = [r.get("chunk_id", "") for r in retrieved]
        retrieved_texts = [r.get("text", "") for r in retrieved]

        context = "\n\n".join(retrieved_texts)

        actual_chunks = self.pipeline.vector_store.get_by_ids(relevant_ids)
        actual_id_to_text = {c["chunk_id"]: c["text"] for c in actual_chunks}
        actual_chunk_texts = [actual_id_to_text.get(cid, "") for cid in relevant_ids]

        result = EvalResult(
            question=question,
            ground_truth=ground_truth,
            generated_answer=answer,
            source_file=item.get("source_file", "unknown"),
            difficulty=item.get("difficulty", "unknown"),
            question_type=item.get("question_type", "unknown"),
            sources_retrieved=retrieved_ids,
            retrieved_chunk_texts=retrieved_texts,
            actual_chunk_ids=relevant_ids,
            actual_chunk_texts=actual_chunk_texts,
            latency_seconds=latency,
        )

        # Retrieval metrics
        result.context_precision = context_precision_at_k(retrieved_ids, relevant_ids, k=5)
        result.context_recall = context_recall_at_k(retrieved_ids, relevant_ids, k=5)
        result.hit_rate = hit_rate(retrieved_ids, relevant_ids, k=5)
        result.mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

        # Answer quality metrics (LLM-judged, sequential)
        faith = self.judge.faithfulness(answer, context)
        result.faithfulness = faith.get("score", 0)
        result.faithfulness_detail = faith

        rel = self.judge.answer_relevance(answer, question)
        result.answer_relevance = rel.get("score", 0)
        result.relevance_detail = rel

        corr = self.judge.correctness(answer, ground_truth, question)
        result.correctness = corr.get("score", 0)
        result.correctness_detail = corr

        return result

    # ── Async mode ─────────────────────────────────────────────────

    def evaluate_async(
        self,
        eval_dataset_path: Path | None = None,
        output_path: Path | None = None,
    ) -> dict:
        """Sync entry point that runs the async evaluation internally."""
        return asyncio.run(self._aevaluate(eval_dataset_path, output_path))

    async def _aevaluate(
        self,
        eval_dataset_path: Path | None = None,
        output_path: Path | None = None,
    ) -> dict:
        """
        Run full evaluation concurrently and produce a report.

        Returns:
            Summary dict with aggregate metrics.
        """
        eval_dataset_path = eval_dataset_path or (self.config.paths.eval_dir / "eval_dataset.json")
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.paths.eval_dir / f"eval_report_{ts}.json"

        if not eval_dataset_path.exists():
            raise FileNotFoundError(
                f"Eval dataset not found at {eval_dataset_path}. "
                "Run `python run_eval.py --generate` first."
            )

        dataset = json.loads(eval_dataset_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(dataset)} evaluation questions")

        # Evaluate all items concurrently (bounded by semaphore)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EVAL)
        tasks = [
            self._aevaluate_single(semaphore, i, item, len(dataset))
            for i, item in enumerate(dataset)
        ]
        results: list[EvalResult] = await asyncio.gather(*tasks)

        summary = self._build_and_save_report(results, output_path)
        return summary

    async def _aevaluate_single(
        self, semaphore: asyncio.Semaphore, idx: int, item: dict, total: int
    ) -> EvalResult:
        """Evaluate a single Q&A pair with concurrent LLM judge calls."""
        async with semaphore:
            question = item["question"]
            ground_truth = item["answer"]
            relevant_ids = item.get("source_chunk_ids", [])

            print(f"\n[{idx + 1}/{total}] {question[:80]}...")

            # RAG query + retrieve (sync pipeline, offloaded to thread)
            loop = asyncio.get_event_loop()
            start = time.time()
            response = await loop.run_in_executor(
                None, lambda: self.pipeline.query(question, use_rerank=True)
            )
            latency = time.time() - start

            answer = response.get("answer", "")
            retrieved = await loop.run_in_executor(
                None, lambda: self.pipeline.retrieve_only(question)
            )
            retrieved_ids = [r.get("chunk_id", "") for r in retrieved]
            retrieved_texts = [r.get("text", "") for r in retrieved]

            context = "\n\n".join(retrieved_texts)

            actual_chunks = self.pipeline.vector_store.get_by_ids(relevant_ids)
            actual_id_to_text = {c["chunk_id"]: c["text"] for c in actual_chunks}
            actual_chunk_texts = [actual_id_to_text.get(cid, "") for cid in relevant_ids]

            result = EvalResult(
                question=question,
                ground_truth=ground_truth,
                generated_answer=answer,
                source_file=item.get("source_file", "unknown"),
                difficulty=item.get("difficulty", "unknown"),
                question_type=item.get("question_type", "unknown"),
                sources_retrieved=retrieved_ids,
                retrieved_chunk_texts=retrieved_texts,
                actual_chunk_ids=relevant_ids,
                actual_chunk_texts=actual_chunk_texts,
                latency_seconds=latency,
            )

            # Retrieval metrics (deterministic, instant)
            result.context_precision = context_precision_at_k(retrieved_ids, relevant_ids, k=5)
            result.context_recall = context_recall_at_k(retrieved_ids, relevant_ids, k=5)
            result.hit_rate = hit_rate(retrieved_ids, relevant_ids, k=5)
            result.mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

            # Run all 3 LLM judge calls concurrently
            faith, rel, corr = await asyncio.gather(
                self.judge.afaithfulness(answer, context),
                self.judge.aanswer_relevance(answer, question),
                self.judge.acorrectness(answer, ground_truth, question),
            )

            result.faithfulness = faith.get("score", 0)
            result.faithfulness_detail = faith
            result.answer_relevance = rel.get("score", 0)
            result.relevance_detail = rel
            result.correctness = corr.get("score", 0)
            result.correctness_detail = corr

            self._log_eval_result(result, prefix=f"  [{idx + 1}]")

            return result

    # ── Shared helpers ─────────────────────────────────────────────

    @staticmethod
    def _log_eval_result(result: EvalResult, prefix: str = " ") -> None:
        """Log answer, ground truth, retrieved/actual chunks, and metrics with color."""
        # ANSI color codes
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RED = "\033[91m"
        DIM = "\033[2m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        # Generated answer vs ground truth
        print(f"{prefix} {CYAN}{BOLD}Generated Answer:{RESET} {result.generated_answer[:200]}")
        print(f"{prefix} {GREEN}{BOLD}Ground Truth:{RESET}     {result.ground_truth[:200]}")

        # Retrieved chunks
        print(f"{prefix} {YELLOW}{BOLD}Retrieved chunks:{RESET}")
        for cid, txt in zip(result.sources_retrieved, result.retrieved_chunk_texts):
            print(f"{prefix}   {YELLOW}-{RESET} {DIM}{cid}{RESET}: {txt[:120]}...")

        # Actual (expected) chunks
        print(f"{prefix} {MAGENTA}{BOLD}Actual chunks:{RESET}")
        for cid, txt in zip(result.actual_chunk_ids, result.actual_chunk_texts):
            preview = f"{txt[:120]}..." if txt else f"{RED}(not found in store){RESET}"
            print(f"{prefix}   {MAGENTA}-{RESET} {DIM}{cid}{RESET}: {preview}")

        # Metrics with color-coded scores
        def _color_score(score: float) -> str:
            if score >= 0.75:
                return f"{GREEN}{score:.2f}{RESET}"
            elif score >= 0.5:
                return f"{YELLOW}{score:.2f}{RESET}"
            else:
                return f"{RED}{score:.2f}{RESET}"

        print(
            f"{prefix} {BOLD}Metrics:{RESET} "
            f"Precision={_color_score(result.context_precision)} "
            f"Recall={_color_score(result.context_recall)} "
            f"Faithful={_color_score(result.faithfulness)} "
            f"Correct={_color_score(result.correctness)} "
            f"{DIM}({result.latency_seconds:.1f}s){RESET}"
        )

    def _build_and_save_report(
        self, results: list[EvalResult], output_path: Path
    ) -> dict:
        """Compute summary, save full report (JSON + CSV), print summary."""
        summary = self._compute_summary(results)

        report = {
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull report saved to {output_path}")

        # Export CSV alongside the JSON report
        csv_path = output_path.with_suffix(".csv")
        self._export_csv(results, csv_path)

        self._print_summary(summary)
        return summary

    @staticmethod
    def _export_csv(results: list[EvalResult], csv_path: Path) -> None:
        """Export evaluation results to a CSV file for easy analysis."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(EvalResult.csv_header())
            for result in results:
                writer.writerow(result.to_csv_row())
        print(f"CSV report saved to {csv_path}")

    @staticmethod
    def _compute_summary(results: list[EvalResult]) -> dict:
        """Compute aggregate metrics across all results."""
        n = len(results)
        if n == 0:
            return {}

        def avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        summary = {
            "total_questions": n,
            "retrieval_metrics": {
                "avg_context_precision": avg([r.context_precision for r in results]),
                "avg_context_recall": avg([r.context_recall for r in results]),
                "avg_hit_rate": avg([r.hit_rate for r in results]),
                "avg_mrr": avg([r.mrr for r in results]),
            },
            "answer_metrics": {
                "avg_faithfulness": avg([r.faithfulness for r in results]),
                "avg_answer_relevance": avg([r.answer_relevance for r in results]),
                "avg_correctness": avg([r.correctness for r in results]),
            },
            "latency": {
                "avg_seconds": avg([r.latency_seconds for r in results]),
                "max_seconds": max(r.latency_seconds for r in results),
                "min_seconds": min(r.latency_seconds for r in results),
            },
            "by_difficulty": {},
            "by_question_type": {},
        }

        # Breakdown by difficulty
        for difficulty in set(r.difficulty for r in results):
            subset = [r for r in results if r.difficulty == difficulty]
            summary["by_difficulty"][difficulty] = {
                "count": len(subset),
                "avg_correctness": avg([r.correctness for r in subset]),
                "avg_faithfulness": avg([r.faithfulness for r in subset]),
            }

        # Breakdown by question type
        for qtype in set(r.question_type for r in results):
            subset = [r for r in results if r.question_type == qtype]
            summary["by_question_type"][qtype] = {
                "count": len(subset),
                "avg_correctness": avg([r.correctness for r in subset]),
                "avg_faithfulness": avg([r.faithfulness for r in subset]),
            }

        return summary

    @staticmethod
    def _print_summary(summary: dict) -> None:
        """Print a formatted summary to stdout."""
        print("\n" + "=" * 60)
        print("  RAG EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\n  Total Questions: {summary['total_questions']}")

        print("\n  Retrieval Metrics:")
        rm = summary["retrieval_metrics"]
        print(f"    Context Precision@5:  {rm['avg_context_precision']:.3f}")
        print(f"    Context Recall@5:     {rm['avg_context_recall']:.3f}")
        print(f"    Hit Rate@5:           {rm['avg_hit_rate']:.3f}")
        print(f"    MRR:                  {rm['avg_mrr']:.3f}")

        print("\n  Answer Quality Metrics:")
        am = summary["answer_metrics"]
        print(f"    Faithfulness:         {am['avg_faithfulness']:.3f}")
        print(f"    Answer Relevance:     {am['avg_answer_relevance']:.3f}")
        print(f"    Correctness:          {am['avg_correctness']:.3f}")

        print("\n  Latency:")
        lat = summary["latency"]
        print(f"    Average:  {lat['avg_seconds']:.2f}s")
        print(f"    Min/Max:  {lat['min_seconds']:.2f}s / {lat['max_seconds']:.2f}s")

        print("\n  By Difficulty:")
        for diff, stats in summary.get("by_difficulty", {}).items():
            print(f"    {diff}: n={stats['count']}, correctness={stats['avg_correctness']:.3f}")

        print("\n  By Question Type:")
        for qtype, stats in summary.get("by_question_type", {}).items():
            print(f"    {qtype}: n={stats['count']}, correctness={stats['avg_correctness']:.3f}")

        print("=" * 60)
