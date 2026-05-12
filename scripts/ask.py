"""
scripts/ask.py — CLI for asking questions.

Usage:
  # Single question
  python scripts/ask.py --question "What optimizer does the Transformer use?"

  # Interactive REPL
  python scripts/ask.py --interactive

  # JSON output (for programmatic use)
  python scripts/ask.py --question "What is BLEU?" --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,   # Quiet for user-facing CLI (change to INFO to debug)
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def ask_question(pipeline_fn, question: str, output_json: bool = False):
    """Run a single question through the pipeline and print the result."""
    response = pipeline_fn(question)

    if output_json:
        # Machine-readable output
        out = {
            "question":      response.question,
            "answer":        response.answer,
            "cannot_answer": response.cannot_answer,
            "was_blocked":   response.was_blocked,
            "citation_count": response.citation_count,
            "citations": [
                {
                    "chunk_number": c.chunk_number,
                    "source":       c.source,
                    "page":         c.page,
                    "preview":      c.preview[:200],
                    "score":        round(c.relevance_score, 4),
                }
                for c in response.citations
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(response.display())


def run_interactive(pipeline_fn):
    """
    Interactive REPL — keeps the pipeline in memory between questions.
    Much faster than re-loading models for each query.
    Type 'quit', 'exit', or Ctrl+C to stop.
    """
    print("\n🔍  Ask My Docs — Interactive Mode")
    print("    Type your question and press Enter.")
    print("    Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        ask_question(pipeline_fn, question)


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your ingested documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ask.py --question "What is scaled dot-product attention?"
  python scripts/ask.py --question "What dataset was used?" --json
  python scripts/ask.py --interactive
        """
    )
    parser.add_argument("--question", "-q", type=str, help="Question to ask.")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start interactive REPL mode.")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (for programmatic use).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show debug logging (pipeline stages, scores, etc.).")
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Build pipeline (loads models into memory — happens once)
    print("⏳  Loading models...", end="", flush=True)
    from src.pipeline import build_pipeline
    try:
        pipeline_fn = build_pipeline()
    except RuntimeError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    print(" ready!\n")

    if args.interactive:
        run_interactive(pipeline_fn)
    elif args.question:
        ask_question(pipeline_fn, args.question, output_json=args.json)


if __name__ == "__main__":
    main()
