# ratchet

AI harness written in Python for local models via LM Studio.

## Currently working

- Terminal UI (Textual) with message input, scrollback log, and `ctrl+l` to clear
- OpenAI-compatible chat completions API client (works with LM Studio and any OpenAI-compatible endpoint)
- Config via `config.toml`, overridable with `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` env vars (auto-loaded from a `.env` file if present — see `.env.example`)
- Offline/error handling when the local server is unreachable or returns an error
- Messages logged to file with timestamps, prefixed by `user:` / `assistant:` role
- Sandboxed shell mode (`uv run ratchet shell`) for running commands scoped to a `./sandbox` directory

See [`docs/model-comparison.md`](docs/model-comparison.md) for comparing the
local model against online models, and
[`docs/shell-commands.md`](docs/shell-commands.md) for the shell mode
commands checklist.

## Setup

```
uv sync
```

## Run

```
uv run ratchet
```

or

```
uv run python -m ratchet
```

## Test

```
uv run pytest
```

## Lint

```
uv run ruff check .
```
