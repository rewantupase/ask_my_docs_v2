"""
scripts/evaluate.py — Offline RAGAS evaluation.

Runs every QA pair in the golden dataset through the pipeline,
collects the answers and retrieved contexts, then scores them with RAGAS.

RAGAS metrics explained:

  faithfulness (most important):
    "Is every claim in the answer actually supported by the retrieved chunks?"
    Steps:
      1. Decompose answer into atomic claims: "BERT uses Adam" → one claim
      2. For each claim, check if it follows from the retrieved chunks
      3. Score = (supported claims) / (total claims)
    Score 1.0 = every sentence is grounded in evidence (no hallucination)
    Score 0.5 = half the claims are hallucinated

  answer_relevancy:
    "Does the answer actually address the question asked?"
    Uses a reverse-generation trick:
      1. Generate N questions from the answer
      2. Score = cosine similarity between generated questions and original
    Catches cases where the model goes off-topic.

  context_precision:
    "Are the retrieved chunks actually relevant to the question?"
    Scores how many of the top-K retrieved chunks are actually useful.
    Low context_precision = retrieval is bringing in noise.

The golden dataset format:
  [
    {
      "question":     "What optimizer does the Transformer use?",
      "ground_truth": "The Transformer uses the Adam optimizer with β1=0.9, β2=0.98, ε=10⁻⁹."
    },
    ...
  ]

CI/CD integration:
  This script exits with code 0 (success) or 1 (failure).
  A GitHub Actions step can check the exit code:
    if [ $? -ne 0 ]; then echo "Quality gate failed!"; exit 1; fi
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_golden_dataset(path: str) -> list:
    """Load and validate the golden QA dataset."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Golden dataset must be a JSON array.")

    required_keys = {"question", "ground_truth"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} missing keys: {missing}")

    print(f"Loaded {len(data)} QA pairs from {path}")
    return data


def run_evaluation(golden_path: str, output_path: str = None):
    """
    Run full RAGAS evaluation on the golden dataset.
    
    Returns the scores dict and whether the quality gate passed.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    # Load golden dataset
    golden = load_golden_dataset(golden_path)

    # Build pipeline
    print("Loading pipeline (models into memory)...")
    from src.pipeline import build_pipeline
    pipeline_fn = build_pipeline()

    # Run every QA pair through the pipeline
    print(f"\nRunning {len(golden)} questions through pipeline...")
    eval_rows = []
    failures = []

    for i, qa in enumerate(golden, 1):
        print(f"  [{i}/{len(golden)}] {qa['question'][:60]}...", end="", flush=True)
        try:
            response = pipeline_fn(qa["question"])
            eval_rows.append({
                "question":     qa["question"],
                "answer":       response.answer,
                "contexts":     [c.preview for c in response.citations]
                               or ["No context retrieved."],
                "ground_truth": qa["ground_truth"],
            })
            status = "✓"
            if response.cannot_answer:
                status = "📭"
            elif response.was_blocked:
                status = "⛔"
            print(f" {status}")
        except Exception as e:
            print(f" ❌ ERROR: {e}")
            failures.append({"question": qa["question"], "error": str(e)})

    print(f"\nSuccessfully processed: {len(eval_rows)}/{len(golden)}")
    if failures:
        print(f"Failed: {len(failures)}")

    if not eval_rows:
        print("No results to evaluate.")
        return None, False

    # Run RAGAS scoring
    print("\nRunning RAGAS evaluation (this calls an LLM internally)...")
    print("Note: RAGAS uses its own LLM for scoring. Set OPENAI_API_KEY if needed,")
    print("or configure RAGAS to use a local model (see ragas docs).\n")

    dataset = Dataset.from_list(eval_rows)
    try:
        scores = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
            ],
        )
    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        print("Tip: RAGAS by default uses OpenAI. To use a free local model:")
        print("  from ragas.llms import LangchainLLMWrapper")
        print("  from langchain_community.llms import Ollama")
        print("  scores = evaluate(..., llm=LangchainLLMWrapper(Ollama(model='llama3.2')))")
        return None, False

    # Format results
    results = {
        "total_questions":   len(golden),
        "evaluated":         len(eval_rows),
        "failed":            len(failures),
        "faithfulness":      round(float(scores["faithfulness"]), 4),
        "answer_relevancy":  round(float(scores["answer_relevancy"]), 4),
        "context_precision": round(float(scores["context_precision"]), 4),
        "thresholds": {
            "faithfulness":      config.MIN_FAITHFULNESS,
            "answer_relevancy":  config.MIN_ANSWER_RELEVANCY,
            "context_precision": config.MIN_CONTEXT_PRECISION,
        },
    }

    # Check quality gate
    gate_passed = (
        results["faithfulness"]      >= config.MIN_FAITHFULNESS        and
        results["answer_relevancy"]  >= config.MIN_ANSWER_RELEVANCY    and
        results["context_precision"] >= config.MIN_CONTEXT_PRECISION
    )
    results["gate_passed"] = gate_passed

    # Print report
    print("\n" + "="*60)
    print("RAGAS EVALUATION REPORT")
    print("="*60)
    print(f"{'Metric':<25} {'Score':>8}  {'Threshold':>10}  {'Status':>8}")
    print("-"*60)
    for metric, threshold_key in [
        ("faithfulness",      "faithfulness"),
        ("answer_relevancy",  "answer_relevancy"),
        ("context_precision", "context_precision"),
    ]:
        score     = results[metric]
        threshold = results["thresholds"][threshold_key]
        status    = "✅ PASS" if score >= threshold else "❌ FAIL"
        print(f"  {metric:<23} {score:>8.4f}  {threshold:>10.2f}  {status:>8}")

    print("-"*60)
    print(f"\nOverall gate: {'✅ PASSED' if gate_passed else '❌ FAILED'}")
    print("="*60)

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results, gate_passed


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on Ask My Docs.",
    )
    parser.add_argument(
        "--golden", type=str,
        default=str(config.EVAL_DIR / "golden_dataset.json"),
        help="Path to golden QA dataset JSON file.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(config.EVAL_DIR / "eval_results.json"),
        help="Where to save results JSON.",
    )
    args = parser.parse_args()

    if not Path(args.golden).exists():
        print(f"Golden dataset not found: {args.golden}")
        print("Create it at eval/golden_dataset.json — see eval/golden_dataset.json.example")
        sys.exit(1)

    _, passed = run_evaluation(args.golden, output_path=args.output)

    # Exit code for CI/CD: 0 = pass, 1 = fail
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
