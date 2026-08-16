# Model comparison: local vs. online

## Local

| Setting  | Value                    |
| -------- | ------------------------ |
| provider | lmstudio                 |
| base_url | http://localhost:1234/v1 |
| model    | qwen/qwen3-4b-2507       |

## OpenRouter candidates

| Model ID                      | Price ($/1M tok, prompt/completion) |
| ----------------------------- | ----------------------------------- |
| `qwen/qwen3-8b`               | 0.117 / 0.455                       |
| `openai/gpt-4o-mini`          | 0.15 / 0.60                         |
| `deepseek/deepseek-chat-v3.1` | 0.25 / 0.95                         |
| `openai/gpt-oss-20b:free`     | 0 / 0                               |

## Flagship models

| Model ID                    | Vendor    | Price ($/1M tok, prompt/completion) |
| --------------------------- | --------- | ----------------------------------- |
| `anthropic/claude-sonnet-5` | Anthropic | 2.00 / 10.00                        |
| `openai/gpt-5.6-terra`      | OpenAI    | 1.00 / 6.00                         |
| `google/gemini-3.5-flash`   | Google    | 1.50 / 9.00                         |
| `x-ai/grok-4.3`             | xAI       | 1.25 / 2.50                         |

## Results (2026-08-16)

Run: `uv run python scripts/compare_models.py`
Local: `qwen/qwen3-4b-2507` (LM Studio) — Flagship: `anthropic/claude-sonnet-5` (OpenRouter)

| #   | Query                         | Category             | Local                  | Flagship              | Local latency | Flagship latency |
| --- | ----------------------------- | -------------------- | ---------------------- | --------------------- | ------------- | ---------------- |
| 1   | Capital of Japan              | Simple Q&A           | correct                | correct               | 0.7s          | 3-6s             |
| 2   | 17 * 24                       | Simple math          | correct                | correct               | 1-4s          | 3-4s             |
| 3   | Countries bordering Germany   | Geography            | wrong, missing 3       | correct               | 3s            | 4s               |
| 4   | Hello world in Python         | Syntax               | correct                | correct               | 2s            | 4s               |
| 5   | git undo commit, keep staged  | Syntax               | correct                | correct               | 10s           | 6s               |
| 6   | One-sentence summary          | Summarization        | correct                | correct               | 2s            | 3s               |
| 7   | Integral of x^2 sin(x)        | Complex math         | correct, no self-check | correct, self-checked | 14-15s        | 6-6.3s           |
| 8   | Instagram bio, no letter e    | Creative constraint  | fails constraint       | passes                | 1-1.5s        | 9-11s            |
| 9   | False-premise arithmetic      | Trick question       | correct                | correct               | 19-25s        | 6-7s             |
| 10  | Farmer/sheep riddle           | Multi-step reasoning | correct                | correct               | 3-4s          | 3-4s             |
| 11  | Explain recursive Fibonacci   | Code understanding   | correct                | correct               | 14-16s        | 6.5-7s           |
| 12  | Who wrote Pride and Prejudice | General knowledge    | correct                | correct               | 0.7-0.8s      | 3.6-5.3s         |

Average latency (12 queries): local 6.6s, flagship 5.3s.
