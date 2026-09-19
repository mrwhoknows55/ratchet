from dataclasses import dataclass, field


@dataclass
class TurnEvent:
    phase: str
    step: int
    index: int = 0
    name: str = ""
    arguments: dict = field(default_factory=dict)
    output: str = ""
    exit_code: int = 0
    elapsed: float = 0.0
    depth: int = 0
    agent: str = ""


@dataclass
class TurnResult:
    text: str
    status: str = "ok"
    steps: int = 0
    elapsed: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
