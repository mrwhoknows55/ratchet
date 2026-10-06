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

Plan mode allows inspection and plan creation, while blocking workspace edits and commands.
Plans are saved inside the sandbox as `PLAN.json` and a readable `PLAN.md` checklist.
Toggle it in the TUI with `Ctrl+B`.

The TUI token meter shows the latest parent request's prompt plus output tokens
against the context window, with a usage bar and percentage.
It turns yellow at 80% and red at 95%.
Session tokens accumulate across requests, including subagents, until memory is reset.
Local LM Studio windows are discovered from the loaded model configuration.
For other endpoints, set `context_window = 32768` to your actual limit under
`[model]` or the selected `[models.<alias>]` entry in `config.toml`.
Unknown limits display as unknown; usage updates when the provider reports tokens.
Picking a model clears the previous context reading while retaining session usage.

The TUI restores the active plan at startup in a scrollable side panel.
Plan creation, retrieval, and status updates refresh the panel and print progress cards in both frontends.
Cards show step IDs, statuses, target files, notes, risks, and the completed-step percentage.
Updates from delegated planners appear alongside parent-agent updates.

Planning can delegate inspection to `researcher` and `planner` subagents.
Planners inspect architecture and save structured plans without executing them.
Only researchers may run in parallel during planning; planners save shared plan files sequentially.
Child turns inherit plan-mode restrictions and enforce their role's tool permissions at runtime.
Delegated agents cannot invoke tools outside their role or spawn further subagents.

`ask_user_question` pauses the agent for a clarifying answer, including inside planner subagents.
The CLI accepts a numbered option or custom text; Ctrl+C cancels the question.
The TUI shows a question dialog with choices and a custom-answer field; Escape cancels.
Answers return to the asking agent's tool history; top-level answers persist with the conversation.
Planner answers stay in the planner's isolated history and inform its returned summary.
Cancelled questions, blank answers, and unavailable interactive handlers return errors without defaults.
Parallel researchers cannot ask interactive questions; they report ambiguities to the parent agent.

`create_plan` accepts string steps or structured steps with IDs, descriptions, target files,
notes, and statuses, plus an optional summary and risks.
`get_plan` retrieves the active plan and completion percentage, including during planning.
During execution, `update_plan_step` records `pending`, `in_progress`, `completed`, `skipped`,
or `failed` statuses and regenerates the checklist.
Omitting notes preserves existing notes; an empty string clears them.
`PLAN.json` is authoritative; use planning tools instead of editing checklist statuses directly.
Existing markdown-only plans must be recreated before structured execution.

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
