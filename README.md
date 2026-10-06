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
- Tool calling in chat mode: 17 tools covering reads, writes, edits, search, file management, snapshots/rollback, shell commands, PATH checks and the web; tool execution always runs locally, regardless of which model backend answers
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
