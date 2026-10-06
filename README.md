# ratchet

AI harness written in Python for local models via LM Studio.

## Currently working

- Terminal UI (Textual) with message input, scrollback log, and `ctrl+l` to clear
- OpenAI-compatible chat completions API client (works with LM Studio and any OpenAI-compatible endpoint)
- Config via `config.toml`, overridable with `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` env vars (auto-loaded from a `.env` file if present — see `.env.example`)
- Offline/error handling when the local server is unreachable or returns an error
- Messages logged to file with timestamps, prefixed by `user:` / `assistant:` role
- Sandboxed shell mode (`uv run ratchet shell`) for running commands scoped to a `./sandbox` directory
- Headless CLI mode: pass a prompt to skip the TUI and print one turn — `uv run ratchet "list the files"` or `uv run ratchet shell "ls"`
- Tool calling in chat mode covers reads, writes, edits, search, file management, snapshots/rollback, shell commands, media, PATH checks and the web; tool execution always runs locally, regardless of which model backend answers
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

Harness support for these: `openpyxl` is a project dependency; media tools
use `yt-dlp`, `ffmpeg`, `ffprobe` and `tesseract` on `PATH`.
`run_command` takes a `timeout` (default
`agent.command_timeout`, capped at 600s) so downloads and encoding are not
killed at 10s. Video OCR is available, but complete benchmark solutions
still require task-specific parsing and verification.

## Media Tools

- `download_video`: accepts `url` and `output_path`, downloading one video
  to MP4 using yt-dlp. Supports YouTube and other yt-dlp-supported sites;
  playlist inputs are limited to one item. Download only content you may access.
- `extract_text`: accepts `path` and `output_path` for image OCR. Set
  `media_type` to `video` for timestamped frame samples. This extracts visible
  text, not speech. Defaults: `language=eng`, `psm=3`, `interval=1` second,
  `start=0`, `max_frames=300`. Optional `end` bounds the video segment;
  `psm=6` is useful for a uniform text block, such as a terminal screen.
- Both use sandbox-relative paths, require existing output parent directories,
  refuse overwrites, and clean up temporary intermediates. Their `timeout`
  defaults to 120 seconds for the entire operation, with a maximum of 600.
- Video OCR permits up to 1,000 frames per call. Read the sampling metadata;
  if capped, continue at `sampled_end + interval` using another output path.
  Sampling can miss brief text; decrease `interval` when necessary.

OCR writes UTF-8 JSON containing `source`, `media_type`, and `records`.
Each record contains `timestamp` (null for images) and the raw recognized
`text`. Video output also contains `sampling` metadata and cap status.
Timestamps represent nominal sampling positions, not exact frame timestamps.
Blank and repeated observations remain intact; command extraction and
deduplication belong to the caller, not the OCR tool.

For example, download to `zork.mp4`, then OCR to `zork-ocr.json` using
`media_type=video` and `psm=6`. Read the artifact before writing
`app/solution.txt`; repeated frames are not necessarily repeated moves.

Install the external binaries separately (macOS: `brew install yt-dlp ffmpeg
tesseract`). FFmpeg supplies `ffprobe`; install additional Tesseract language
data if needed. Check prerequisites using `check_command`.
No new Python dependency or cloud OCR service is required.

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

or headless, one prompt at a time:

```
uv run ratchet "list the files"
uv run ratchet shell "ls"
```

Headless mode persists conversation memory across process calls, to `log/session.json`:

```
uv run ratchet "My name is Sam."
uv run ratchet "What is my name?"
> Your name is Sam.

# wipe stored session memory (any of the three works)
uv run ratchet --new "What is my name?"
uv run ratchet --clear
uv run ratchet --reset
```

Plan mode is read-only: the agent inspects the sandbox and saves a checklist to `PLAN.md`.
Toggle it in the TUI with `Ctrl+B`.

```
uv run ratchet --plan "Design a math module with safe division and caching"
uv run ratchet --execute-plan
```

## Test

```
uv run pytest
```

## Lint

```
uv run ruff check .
```
