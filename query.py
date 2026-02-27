"""
Interactive query interface for the RAG system.

Usage:
    python query.py "What is the total revenue for 2024?"
    python query.py --interactive          # Start interactive session
    python query.py --chat                 # Continuous chat with history
    python query.py --no-rerank "question" # Disable reranking
"""

import argparse
import json
import logging
import sys

from config import get_config
from rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--chat", "-c", action="store_true", help="Continuous chat mode with conversation history")
    parser.add_argument("--no-rerank", action="store_true", help="Disable LLM reranking")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    pipeline = RAGPipeline(config)

    if pipeline.vector_store.count == 0:
        print("No documents in vector store. Run `python ingest.py` first.")
        sys.exit(1)

    print(f"Vector store: {pipeline.vector_store.count} chunks loaded\n")

    use_rerank = not args.no_rerank

    if args.interactive:
        _interactive_mode(pipeline, use_rerank, args.json)
    elif args.chat:
        _chat_mode(pipeline, use_rerank, args.json)
    elif args.question:
        _single_query(pipeline, args.question, use_rerank, args.json)
    else:
        parser.print_help()
        sys.exit(1)


def _single_query(pipeline: RAGPipeline, question: str, use_rerank: bool, raw_json: bool):
    """Process a single question."""
    result = pipeline.query(question, use_rerank=use_rerank)

    if raw_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_result(result)


def _interactive_mode(pipeline: RAGPipeline, use_rerank: bool, raw_json: bool):
    """Run an interactive Q&A session."""
    print("RAG Interactive Mode (type 'quit' to exit)\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        result = pipeline.query(question, use_rerank=use_rerank)

        if raw_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _print_result(result)

        print()


def _chat_mode(pipeline: RAGPipeline, use_rerank: bool, raw_json: bool):
    """Run a continuous chat session that maintains conversation history.

    Special commands:
        /clear   - Clear conversation history
        /history - Show conversation history
        quit     - Exit the session
    """
    print("RAG Continuous Chat Mode")
    print("  Commands: /clear (reset history), /history (show history), quit (exit)")
    print(f"  Reranking: {'on' if use_rerank else 'off'}\n")

    chat_history: list[dict] = []
    turn_number = 0

    while True:
        try:
            question = input(f"[{turn_number + 1}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        # Handle special commands
        if question == "/clear":
            chat_history.clear()
            turn_number = 0
            print("Conversation history cleared.\n")
            continue

        if question == "/history":
            if not chat_history:
                print("No conversation history yet.\n")
            else:
                print("\n" + "─" * 60)
                print("Conversation History:")
                print("─" * 60)
                for msg in chat_history:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    content = msg["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"  {role}: {content}")
                print("─" * 60 + "\n")
            continue

        result = pipeline.query_with_history(
            question,
            chat_history=chat_history,
            use_rerank=use_rerank,
        )

        if raw_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _print_result(result)

        # Append this turn to history
        answer = result.get("answer", "")
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        turn_number += 1

        # Keep history bounded to avoid token overflow (last 20 turns = 40 messages)
        max_messages = 40
        if len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]

        print()


def _print_result(result: dict):
    """Pretty-print a query result with color."""
    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{DIM}{'─' * 60}{RESET}")
    print(f"{GREEN}{BOLD}Answer:{RESET}")
    print(f"{GREEN}{result.get('answer', 'No answer')}{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")

    sources = result.get("sources", [])
    if sources:
        print(f"\n{YELLOW}{BOLD}Sources:{RESET}")
        for s in sources:
            score = s.get("score", 0)
            # Color score based on value
            if score >= 0.7:
                score_color = GREEN
            elif score >= 0.4:
                score_color = YELLOW
            else:
                score_color = MAGENTA
            print(
                f"  {CYAN}-{RESET} {s['source_file']} {DIM}|{RESET} "
                f"{s['section']} {DIM}(score: {RESET}{score_color}{score:.3f}{RESET}{DIM}){RESET}"
            )

    usage = result.get("usage", {})
    if usage:
        print(f"\n{DIM}Tokens: {usage.get('total_tokens', 'N/A')}{RESET}")


if __name__ == "__main__":
    main()
