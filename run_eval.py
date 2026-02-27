"""
Evaluation runner entry point.

Usage:
    python run_eval.py --generate                # Generate eval dataset (sync)
    python run_eval.py --evaluate                # Run evaluation (sync)
    python run_eval.py --generate --evaluate     # Generate + evaluate (sync)
    python run_eval.py --generate --async        # Generate eval dataset (async, faster)
    python run_eval.py --generate --evaluate --async  # Both in async mode
"""

import argparse
import logging
import sys

from config import get_config
from eval.evaluator import RAGEvaluator
from eval.generate_eval_data import EvalDataGenerator


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate eval Q&A dataset")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async mode for faster concurrent execution",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions per document (default: from config)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if not args.generate and not args.evaluate:
        parser.print_help()
        print("\nSpecify --generate, --evaluate, or both.")
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    if args.num_questions is not None:
        from dataclasses import replace

        new_eval = replace(config.eval, num_questions_per_doc=args.num_questions)
        config = replace(config, eval=new_eval)

    mode = "async" if args.use_async else "sync"

    if args.generate:
        print("=" * 60)
        print(f"  GENERATING EVALUATION DATASET  [{mode}]")
        print("=" * 60)
        generator = EvalDataGenerator(config)
        if args.use_async:
            dataset = generator.generate_eval_dataset_async()
        else:
            dataset = generator.generate_eval_dataset()
        print(f"\nGenerated {len(dataset)} Q&A pairs\n")

    if args.evaluate:
        print("=" * 60)
        print(f"  RUNNING RAG EVALUATION  [{mode}]")
        print("=" * 60)
        evaluator = RAGEvaluator(config)
        if args.use_async:
            summary = evaluator.evaluate_async()
        else:
            summary = evaluator.evaluate()


if __name__ == "__main__":
    main()
