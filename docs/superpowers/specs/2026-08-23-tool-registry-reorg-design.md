# Tool registry reorg, step limit, and loader UX

Date: 2026-08-23

## Problem

`agent/tools.py` currently mixes three unrelated concerns in one file: the
tool schema/dispatch registry, the agentic tool-call loop (`run_agent_turn`),
and a hardcoded iteration cap (`MAX_TOOL_ITERATIONS = 5`). Tool-call progress
is reported to the UI as plain strings ("tool: X running...", "tool: X ->
result"), with no structured success/failure or timing info, and no visual
indication in the TUI beyond static log lines.

This is the first of three planned changes to the tool system (edit-file tool
and paginated read follow later as their own specs). This one is
foundational: it sets the module layout and the step-reporting interface the
other two will build on.

## Goals

- Split tool registry code from loop code, and organize tools by relation
  (file ops today; room for future categories).
- Replace the hardcoded step cap with a `config.toml` value.
- Report each tool call to the UI as structured data (step count, tool,
  args, success/failure, duration), not a hand-formatted string.
- Show a live spinner in the TUI while a tool is running, replacing the
  static "running..." log line.

## Non-goals

- No loader while waiting on the LLM itself — only during tool execution.
- No output preview in the step display.
- No new tools (edit_file, paginated read) — separate specs.

## Design

### Module layout

```
src/ratchet/agent/
  loop.py            # run_agent_turn moves here, replacing the unused
                      # run() stub (and its tests, which get deleted)
  tools/
    __init__.py       # aggregates TOOL_SCHEMAS + execute_tool() dispatch
    file_ops.py        # list/read/write/delete/search: schemas + execute fns
```

`file_ops.py` is the only category today. A future category (e.g. shell-exec
tools) becomes its own `tools/<category>_ops.py` plus one import line in
`tools/__init__.py`. `edit_file` (planned next) is a file-relation tool, so
it adds an entry to `file_ops.py` rather than a new file.

### Tool registry

Each category module exports:

```python
SCHEMAS: list[dict]
EXECUTORS: dict[str, Callable[[dict, Path], dict]]  # -> {"output": str, "success": bool}
```

`tools/__init__.py` merges these across categories:

```python
TOOL_SCHEMAS = [*file_ops.SCHEMAS]
_EXECUTORS = {**file_ops.EXECUTORS}

def execute_tool(name: str, arguments: dict, sandbox_root: Path) -> dict:
    executor = _EXECUTORS.get(name)
    if executor is None:
        return {"output": f"Error: unknown tool '{name}'", "success": False}
    return executor(arguments, sandbox_root)
```

`success` comes from the underlying `exit_code` in `shell/executor.py`'s
return dicts (already computed, currently discarded by `execute_tool`):
`exit_code == 0` for list/read/write/delete. `search_files` is the one
exception — `rg`/`grep` return exit code `1` for "no matches found" (not an
error), so its wrapper treats `exit_code in (0, 1)` as success and anything
higher (e.g. invalid pattern) as failure.

### Step limit config

`config.toml` gets a new section:

```toml
[agent]
max_steps = 5
```

`DEFAULT_CONFIG` in `agent/config.py` gets a matching `"agent": {"max_steps":
5}` entry. `run_agent_turn` reads `load_config()["agent"]["max_steps"]`
(falling back to 5), the same pattern `client.py` already uses for model
config. One step = one LLM round-trip; however many tool calls that
round-trip requests all count toward that one step.

### Structured step events

```python
@dataclass
class ToolStepEvent:
    step: int
    max_steps: int
    tool_name: str
    arguments: dict
    phase: Literal["start", "end"]
    success: bool | None = None      # set on "end" only
    duration_ms: float | None = None # set on "end" only
```

`run_agent_turn` gains an `on_tool_call: Callable[[ToolStepEvent], None] |
None` parameter (replacing the current string-based one), firing once per
tool call at `phase="start"` (before execution) and once at `phase="end"`
(after, with `success`/`duration_ms` filled in).

`format_tool_event(event: ToolStepEvent) -> str` (in `loop.py`) renders the
default plain-text form, used both for the TUI's scrollback line and the log
file:

```
[2/5] read_files(path="a.txt") running...
[2/5] read_files(path="a.txt") -> ok (12ms)
[3/5] delete_file(path="missing.txt") -> FAILED (3ms)
```

No output preview, no color/markup — same plain text in both the TUI log and
the log file.

### Loader UX (TUI)

A new `ToolStatusIndicator` widget (subclasses Textual's `Static`, renders a
`rich.spinner.Spinner`) sits between the `RichLog` and the `Input` in
`RatchetApp.compose()`. Hidden by default.

- `phase="start"`: indicator shows the formatted "running..." line and
  becomes visible.
- `phase="end"`: indicator hides; the formatted result line is appended to
  the `RichLog` scrollback and the log file (the "running..." line is never
  written to the scrollback/log, only shown transiently in the indicator).

Loader is scoped to tool execution only — no spinner while waiting on the
LLM's own response.

## Test fallout

- `tests/test_agent_loop.py`: delete the existing tests for the dead `run()`
  stub; add tests for `run_agent_turn` (moved from `test_agent_tools.py`),
  updated for the new `dict` return shape from `execute_tool` and the
  `ToolStepEvent`-based callback. Add a test for `max_steps` config wiring
  (default 5, overridden via `config.toml`) and one for `format_tool_event`
  output for both success and failure cases.
- `tests/test_agent_tools.py`: keep the registry-level tests (schemas,
  `execute_tool` per tool, unknown-tool case), updated for the new
  `{"output", "success"}` return shape.
- `tests/test_tui_main.py`: update the two existing tool-call tests to
  assert on the new loader/scrollback behavior instead of the old
  "running..."/"->" string pair.

## Call sites

- `tui/main.py`: import `run_agent_turn` from `ratchet.agent.loop` instead of
  `ratchet.agent.tools`; add the `ToolStatusIndicator` widget; replace
  `_on_tool_call(message: str)` with a handler that takes a `ToolStepEvent`,
  drives the indicator, and writes `format_tool_event(event)` to the log on
  `phase="end"`.
