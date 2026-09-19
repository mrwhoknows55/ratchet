from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class AgentContext:
    sandbox_root: Path
    call_llm_fn: Callable | None = None
    override_config: dict | None = None
    on_event: Callable | None = None
    depth: int = 0


def as_context(value: "AgentContext | Path") -> "AgentContext":
    if isinstance(value, AgentContext):
        return value
    return AgentContext(sandbox_root=value)
