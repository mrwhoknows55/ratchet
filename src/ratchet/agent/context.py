from dataclasses import dataclass
from pathlib import Path
from typing import Callable

QuestionHandler = Callable[[str, list[str]], str | None]


@dataclass(frozen=True)
class AgentContext:
    sandbox_root: Path
    call_llm_fn: Callable | None = None
    override_config: dict | None = None
    on_event: Callable | None = None
    depth: int = 0
    plan_mode: bool = False
    allowed_tools: frozenset[str] | None = None
    ask_user: QuestionHandler | None = None
    plan_root: Path | None = None


def as_context(value: "AgentContext | Path") -> "AgentContext":
    if isinstance(value, AgentContext):
        return value
    return AgentContext(sandbox_root=value)
