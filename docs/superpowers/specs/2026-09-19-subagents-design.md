# Subagents: delegation, roles, and nested events

Date: 2026-09-19

## Problem

Every task in ratchet runs in one conversation. A long lookup — finding where
a symbol lives, reading three candidate files, searching the web for a flag —
lands its full tool output in the parent's message history and stays there for
the rest of the session. The parent pays for that context on every subsequent
step.

Delegation fixes this: a focused, isolated agent does the digging and returns
a summary. `hydraharness/src/07-harness` has a working version of the idea,
but its implementation fights ratchet's existing contracts. It duplicates the
whole reasoning loop into `subagent.py`, passes its event callback through a
module-level global, and returns a raw result dict that has no relationship to
ratchet's uniform `{stdout, stderr, exit_code}` tool contract.

This spec ports the idea, not the code.

## Goals

- One blocking subagent per call, with an isolated message history.
- Four roles, each restricted to a subset of ratchet's tools.
- Subagents run the same loop the parent runs, differently configured.
- Nested tool calls are visible live in both the CLI and the TUI.
- The parent receives a summary plus consumption metadata, never a transcript.
- Leave parallel fan-out a narrow, cheap change rather than a rewrite.

## Non-goals

- No parallelism, no batch tool, no dependency graph. That is v2.
- No sandbox isolation between agents — they share one root.
- No nesting: `max_depth` is 1, so subagents cannot spawn subagents.
- No new provider code. Model selection routes through `override_config`.

## Design

### Module layout

```
src/ratchet/agent/
  context.py    AgentContext                      (new)
  subagent.py   ROLE_TOOLS, role prompts,
                spawn_subagent                    (new)
  loop.py       run_agent_turn, TurnEvent,
                TurnResult                        (replaces the echo stub)
  tools.py      TOOL_SCHEMAS, _dispatch,
                execute_tool_result, schema_for_role
  client.py     call_llm, now capturing usage
  config.py     [subagent] defaults
prompts/
  system.md     unchanged, parent only
  roles/        researcher.md, coder.md,
                tester.md, generalist.md          (new)
```

Moving `run_agent_turn` into `loop.py` completes what spec 2026-08-23 already
specified and never landed. The existing `loop.py` echo stub and its four
tests in `tests/test_agent_loop.py` are deleted, as that spec called for.

Import cycle: `tools` needs `spawn_subagent`, `subagent` needs
`run_agent_turn`, `loop` needs `_dispatch`. Broken with one function-level
import of `run_agent_turn` inside `spawn_subagent`, the same place
hydraharness breaks it.

### AgentContext

A frozen dataclass replacing the bare `sandbox_root` parameter through the
dispatch layer:

```python
@dataclass(frozen=True)
class AgentContext:
    sandbox_root: Path
    call_llm_fn: Callable
    override_config: dict | None = None
    on_event: Callable[[TurnEvent], None] | None = None
    depth: int = 0
```

`run_agent_turn` builds one per turn. `_dispatch` takes it in place of
`sandbox_root`; the sixteen existing tools read `ctx.sandbox_root` and ignore
the rest. `spawn_subagent` reads all five fields. The edit is wide and
mechanical — no tool changes behavior.

### Generalized run_agent_turn

The public signature keeps its current positional parameters so `cli.py` and
`tui/main.py` are unaffected on the way in, and gains keyword-only ones:

```python
def run_agent_turn(
    call_llm_fn, user_text, sandbox_root,
    override_config=None, on_event=None, messages=None,
    *,
    tools_schema=None,      # defaults to TOOL_SCHEMAS
    system_prompt=None,     # defaults to prompts/system.md
    max_steps=None,         # defaults to agent.max_steps
    depth=0,
    agent="",
) -> TurnResult
```

Parent and subagent are the same function. A subagent is the role's schema,
the role's prompt, a lower step budget, and a depth of one.

### TurnResult

`run_agent_turn` returns a dataclass rather than a bare string, because the
metadata the parent needs is only visible from inside the loop:

```python
@dataclass
class TurnResult:
    text: str
    status: str          # "ok" | "max_steps" | "error"
    steps: int
    elapsed: float
    tools_used: list[str]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
```

`cli.py` and `tui/main.py` use `.text` where they used the return value
directly. Two call sites.

### Token capture

`call_llm` currently reads `choices[0].message` and discards the rest. It
gains a `usage` key in its result dict when the response carries one, dropped
silently when it does not — LM Studio returns it, not every provider does.
`run_agent_turn` accumulates across its own steps only; a subagent's tokens
are reported by the subagent, not folded into the parent's total.

### Roles

A role is a name mapped to a list of ratchet tool names.

| role | tools |
|---|---|
| researcher | list_files, search_files, file_search, read_files, read_file_range, get_file_info, check_command, search_web, fetch_url |
| coder | the six read tools, plus write_files, replace_in_file, append_file, delete_file, copy_file, move_file, rollback_file |
| tester | list_files, search_files, file_search, read_files, read_file_range, get_file_info, run_command, check_command |
| generalist | all sixteen, minus spawn_subagent |

`generalist` is the default when the model omits a role. Coder gets no web
and no shell; tester gets no writes; researcher gets neither.

`schema_for_role(role)` filters `TOOL_SCHEMAS` by that list. `spawn_subagent`
is never in a subagent's schema. Dispatch independently refuses it when
`ctx.depth >= subagent.max_depth`, so a model that hallucinates the tool gets
a clear error rather than infinite recursion.

### Role prompts

Four self-contained files under `prompts/roles/`. They do not inherit
`system.md` — each states its own remit, its tool limits, and the requirement
to end with a summary written for the parent agent rather than for a user.
Kept short, so the duplicated grounding rules have a small surface to drift
across.

### The spawn_subagent tool

Schema parameters: `task` (required), `role` (enum, default `generalist`),
`context` (optional background from the parent), `max_steps` (optional, may
lower the configured budget, never raise it).

Execution: resolve the role, load its prompt, filter the schema, build a fresh
two-message history of role prompt plus task-and-context, call
`run_agent_turn` at `depth + 1` with the same event sink, and pack the
`TurnResult` into ratchet's tool contract.

### Result contract

`stdout` is the subagent's own summary followed by one metadata line:

```
[researcher · 5/8 steps · 4.1s · 1240 tok · read_files, search_files]
```

The token field is omitted when the provider reported no usage.

| outcome | stdout | stderr | exit |
|---|---|---|---|
| finished | summary + metadata | empty | 0 |
| budget exhausted | partial summary + metadata + "stopped at N steps" | empty | 1 |
| hard failure | empty | reason | 1 |

The parent's history gains exactly one tool message. The nested turns are
never serialized into it — the parent already watched them scroll past.

### Events

`TurnEvent` gains two fields, both defaulted so existing parent events are
untouched:

```python
depth: int = 0
agent: str = ""
```

Formatters indent by `depth` and prefix `[{agent}]` when `agent` is set. The
TUI status bar keeps showing the parent's step and appends the subagent's own
step while one is running; `_on_turn_event` retains the last depth-0 step to
do this. The CLI indents identically, without panels.

### Config

```toml
[subagent]
max_steps = 8
max_depth = 1

[subagent.models]
researcher = ""   # empty means inherit the parent's model
coder      = ""
tester     = ""
generalist = ""
```

A non-empty value names an existing `[models.*]` alias and is applied via
`override_config`, the same path the TUI model picker already uses. No
per-provider code, consistent with `call_llm` staying OpenAI-compatible.

`DEFAULT_CONFIG` in `config.py` gains matching defaults.

## Test fallout

- `tests/test_subagent.py` (new): role filters the schema; nesting refused at
  depth; budget exhaustion exits 1 with a partial summary; hard failure puts
  the reason in stderr; metadata line reflects real steps, tools and tokens;
  nested events carry depth and agent.
- `tests/test_agent_client.py`: usage captured when present, absent key
  tolerated.
- `tests/test_agent_tools.py`: dispatch takes `AgentContext`; registry tests
  otherwise unchanged. Loop tests move to `test_agent_loop.py`.
- `tests/test_agent_loop.py`: delete the echo-stub tests; house the
  `run_agent_turn` tests, updated for `TurnResult`.
- `tests/test_tui_main.py`: `.text` instead of the bare return; depth
  indentation and agent label in the formatters.

The fake `call_llm_fn` already used throughout `test_agent_tools.py` drives
subagents too, so none of this needs a live server.

## Build order

Each step leaves the suite green.

1. Capture `usage` in `client.py`.
2. Add `AgentContext`; thread it through `_dispatch`.
3. Move the loop to `loop.py`; generalize it; return `TurnResult`.
4. Add `depth` and `agent` to `TurnEvent`; indent in CLI and TUI.
5. Role table and the four prompt files.
6. `spawn_subagent`, its schema, and the `[subagent]` config.

## What v2 needs

A batch tool — tasks plus a mode — and a thread pool over step 6's single
recursive call. The context object, the depth field and the summary-only
contract are what keep that change small. Shared-sandbox write conflicts are
deferred until then, and are the reason parallel mode will likely be
restricted to read-only roles.
