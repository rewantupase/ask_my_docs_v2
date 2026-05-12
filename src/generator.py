"""
src/generator.py — LLM answer generation.

Uses Ollama to run Mistral 7B (v0.3) locally on your machine.
Completely free — no API key, no rate limits, no data leaving your machine.

Why Mistral 7B?
  - 7B parameters with exceptional instruction following at this size class
  - Mistral uses Grouped Query Attention (GQA) → faster inference vs LLaMA 7B
  - Sliding Window Attention (SWA) → handles long contexts efficiently
  - 32K context window → can fit many retrieved chunks comfortably
  - Outperforms LLaMA 2 13B on most benchmarks despite being smaller
  - ~4GB RAM with 4-bit quantization (Q4_0) — runs well on any modern laptop
  - Excellent at following structured prompts (required for citation format)
  - 29M+ downloads on Ollama — well-tested and stable

Why Ollama?
  - One-command install, one-command model pull
  - Runs quantized models (4-bit) — Mistral 7B fits in ~4GB RAM
  - HTTP API identical to OpenAI's: easy to swap in other models
  - Supports GPU acceleration (CUDA, Metal) automatically if available
  - Auto-manages model files in ~/.ollama/

Prompt versioning:
  Prompts are loaded from prompts/v1.yaml (tracked by git).
  This means every prompt change is a commit — full audit trail.
  To A/B test prompts: change PROMPT_VERSION in config.py, run eval.py,
  compare faithfulness scores, pick the winner.

Mistral-specific notes:
  - Uses [INST] ... [/INST] instruction format internally (Ollama handles this)
  - Temperature 0.1 keeps answers factual and deterministic
  - stop tokens prevent the model from roleplaying "Human:" turns
"""

import logging
import yaml
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    """Load and cache the prompt template YAML."""
    prompt_file = config.PROMPTS_DIR / f"{config.PROMPT_VERSION}.yaml"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            f"Expected: {config.PROMPTS_DIR}/{config.PROMPT_VERSION}.yaml"
        )
    with open(prompt_file) as f:
        prompts = yaml.safe_load(f)
    logger.info(f"Loaded prompts from {prompt_file} (version {prompts.get('version', '?')})")
    return prompts


def get_rag_prompt(context: str, question: str) -> str:
    """Fill the RAG prompt template with context and question."""
    prompts = _load_prompts()
    return prompts["rag_prompt"].format(context=context, question=question)


def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using the local Ollama LLM.
    
    The prompt instructs the model to:
      1. Only use the provided context chunks
      2. Cite every claim with [Chunk N]
      3. Say "I cannot answer" if information is missing

    Returns the raw LLM response string (citations are parsed by citations.py).
    
    Raises:
        ConnectionError: if Ollama is not running (run: ollama serve)
        RuntimeError: if the model is not pulled (run: ollama pull llama3.2)
    """
    try:
        import ollama
    except ImportError:
        raise ImportError("pip install ollama")

    prompt = get_rag_prompt(context, question)

    logger.info(f"Generating answer via Ollama ({config.OLLAMA_MODEL})...")

    try:
        response = ollama.generate(
            model=config.OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
                "num_gpu": 0,
                "stop": ["Human:", "User:", "Question:"],   # prevent run-on
            },
        )
    except Exception as e:
        if "connection refused" in str(e).lower():
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running?\n"
                "Start it with: ollama serve\n"
                f"Original error: {e}"
            )
        if "model not found" in str(e).lower():
            raise RuntimeError(
                f"Model '{config.OLLAMA_MODEL}' not found.\n"
                f"Pull it with: ollama pull mistral\n"
                f"Original error: {e}"
            )
        raise

    raw = response["response"].strip()
    logger.info(f"Generated {len(raw.split())} word answer.")
    return raw
