# ratchet

AI harness written in Python for local models via LM Studio.

## Currently working

- Terminal UI (Textual) with message input, scrollback log, and `ctrl+l` to clear
- OpenAI-compatible chat completions API client (works with LM Studio and any OpenAI-compatible endpoint)
- Config via `config.toml`, overridable with `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` env vars (auto-loaded from a `.env` file if present — see `.env.example`)
- Offline/error handling when the local server is unreachable or returns an error
- Messages logged to file with timestamps, prefixed by `user:` / `assistant:` role
- Sandboxed shell mode (`uv run ratchet shell`) for running commands scoped to a `./sandbox` directory
- Tool calling in chat mode: 16 tools covering reads, writes, edits, search, file management, snapshots/rollback, shell commands and the web; tool execution always runs locally, regardless of which model backend answers
- Web tools (`search_web`, `fetch_url`) via the Tavily REST API — set `TAVILY_API_KEY` to enable them; everything else works without it

See [`docs/model-comparison.md`](docs/model-comparison.md) for comparing the
local model against online models, and
[`docs/shell-commands.md`](docs/shell-commands.md) for the shell mode
commands checklist.

## Benchmark tasks

<details>
<summary><strong>Tasks to be benchmarked on (taken from <code>terminal-bench</code>)</strong></summary>

- [working] Create a file called `hello.txt` in the current directory. Write "Hello, world!" to it. Make sure it ends in a newline. Don't make any other files or folders.
- [working] Extract the contents of the solution from `archive.tar` and write it to `app/solution.txt`.
- [working] Instructions:
  1. Create a new spreadsheet named "Financial Report"
  2. Create a sheet in this spreadsheet named "Q1 Data"
  3. Add the following data to the sheet, starting at cell A1:

     | Month    | Revenue | Expenses |
     | -------- | ------- | -------- |
     | January  | 10000   | 8000     |
     | February | 12000   | 9000     |
     | March    | 15000   | 10000    |

  4. Add a third column that contains the profit.
- [wip] Download this video of someone playing zork: https://www.youtube.com/watch?v=ZCbvyPbhRfA. Then transcribe the entire contents of the text, and create a file `app/solution.txt` that has all the moves they input, one per line, in the format `n` or `get bag` etc.
- [wip] Download the first video ever uploaded to YouTube as an mp4. Then, trim the video to the final 10 seconds and save it as `result.mp4`.

</details>

Harness support for these: `openpyxl` is a project dependency, `yt-dlp` and
`ffmpeg` are expected on `PATH`, and `run_command` takes a `timeout` (default
`agent.command_timeout`, capped at 600s) so downloads and encoding are not
killed at 10s. Transcription has no tool yet — the zork task is blocked on it.

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
