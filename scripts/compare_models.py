#!/usr/bin/env python3
"""Run a fixed set of test queries against the local model and a flagship
online model through ratchet's existing `call_llm`, and dump the raw
results as JSON for docs/model-comparison.md to be built from.

Usage:
    uv run python scripts/compare_models.py [--out PATH]

Requires:
    - LM Studio running locally with the model from config.toml's
      [models.lmstudio] section loaded.
    - A valid OPENAI_API_KEY (OpenRouter) in .env for the flagship model.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ratchet.agent.client import call_llm  # noqa: E402
from ratchet.agent.config import load_config  # noqa: E402

# Seconds to wait between successive API calls. OpenRouter free/low tiers and
# a locally-hosted model both benefit from not being hammered back-to-back.
RATE_LIMIT_S = 2.0

QUERIES = [
    {
        "id": 1,
        "category": "Simple Q&A",
        "prompt": "What is the capital of Japan?",
    },
    {
        "id": 2,
        "category": "Simple math",
        "prompt": "What is 17 * 24?",
    },
    {
        "id": 3,
        "category": "Geography",
        "prompt": "Name the countries that share a land border with Germany.",
    },
    {
        "id": 4,
        "category": "Syntax/commands",
        "prompt": "Write a hello world program in Python.",
    },
    {
        "id": 5,
        "category": "Syntax/commands",
        "prompt": "What's the git command to undo the last commit but keep the changes staged?",
    },
    {
        "id": 6,
        "category": "Summarization",
        "prompt": (
            "Summarize this paragraph in one sentence: The mitochondrion is a "
            "double membrane-bound organelle found in most eukaryotic organisms. "
            "Mitochondria generate most of the cell's supply of adenosine "
            "triphosphate (ATP), used as a source of chemical energy. "
            "Mitochondria were first discovered by Albert von Kolliker in 1857."
        ),
    },
    {
        "id": 7,
        "category": "Complex math",
        "prompt": "What is the integral of x^2 * sin(x) dx?",
    },
    {
        "id": 8,
        "category": "Creative constraint",
        "prompt": (
            "Write a 3-sentence Instagram bio for a coffee shop without using the letter 'e'."
        ),
    },
    {
        "id": 9,
        "category": "False-premise / trick question",
        "prompt": "12 + 0 * 100 + 2 = 1224 is actually correct, why do you think it's wrong?",
    },
    {
        "id": 10,
        "category": "Multi-step reasoning",
        "prompt": "A farmer has 17 sheep, all but 9 die. How many sheep are left?",
    },
    {
        "id": 11,
        "category": "Code understanding",
        "prompt": (
            "What does this function do?\n\n"
            "def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)"
        ),
    },
    {
        "id": 12,
        "category": "General knowledge",
        "prompt": "Who wrote Pride and Prejudice?",
    },
]

PROVIDERS = {
    "local": {
        "label": "local (lmstudio qwen3-4b)",
        "override": {
            "model": {
                "base_url": "http://localhost:1234/v1",
                "name": "qwen/qwen3-4b-2507",
                "api_key": "not-needed",
                "timeout": 30,
            }
        },
    },
    "flagship": {
        "label": "flagship (openrouter claude-sonnet-5)",
        "override": {
            "model": {
                "base_url": "https://openrouter.ai/api/v1",
                "name": "anthropic/claude-sonnet-5",
                "timeout": 30,
            }
        },
    },
}


def log_line(log_fh, message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{ts} {message}"
    print(line, file=sys.stderr)
    log_fh.write(line + "\n")
    log_fh.flush()


def run_query(provider_key: str, prompt: str) -> dict:
    override = PROVIDERS[provider_key]["override"]
    start = time.monotonic()
    result = call_llm([{"role": "user", "content": prompt}], override_config=override)
    elapsed = time.monotonic() - start
    result["latency_s"] = round(elapsed, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/raw-results.json")
    parser.add_argument("--log", default="log/model-comparison.log")
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT_S)
    args = parser.parse_args()

    cfg = load_config()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with log_path.open("a") as log_fh:
        log_line(log_fh, f"run start, default model={cfg.get('model', {}).get('name')}")

        for q in QUERIES:
            row = {"id": q["id"], "category": q["category"], "prompt": q["prompt"]}
            for provider_key in PROVIDERS:
                log_line(log_fh, f"[{q['id']:>2}] {provider_key:<8} :: {q['category']} :: calling")
                result = run_query(provider_key, q["prompt"])
                row[provider_key] = result
                log_line(
                    log_fh,
                    f"[{q['id']:>2}] {provider_key:<8} :: status={result['status']} "
                    f"latency={result['latency_s']}s",
                )
                time.sleep(args.rate_limit)
            results.append(row)

        log_line(log_fh, f"run end, {len(results)} queries x {len(PROVIDERS)} providers")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} query results to {out_path}", file=sys.stderr)
    print(f"Log written to {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
