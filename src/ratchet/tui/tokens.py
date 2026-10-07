from rich.text import Text
from textual.widgets import Static

from ratchet.agent.events import TurnEvent


class TokenMeter(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__("", markup=False, **kwargs)
        self.context_tokens: int | None = None
        self.context_window: int | None = None
        self.session_tokens: int | None = None
        self.tooltip = (
            "Latest parent request: prompt + output tokens. "
            "Session usage sums all requests, including subagents."
        )

    def record(self, event: TurnEvent) -> None:
        used = (event.prompt_tokens or 0) + (event.completion_tokens or 0)
        self.session_tokens = (self.session_tokens or 0) + used
        if event.depth == 0:
            self.context_tokens = used
            window = event.context_window
            self.context_window = window if type(window) is int and window > 0 else None
        self.update(self.render_tokens())

    def reset_context(self, window: int | None = None) -> None:
        self.context_tokens = None
        self.context_window = window if type(window) is int and window > 0 else None
        self.update(self.render_tokens())

    def reset_session(self) -> None:
        self.session_tokens = None
        self.reset_context(self.context_window)

    def render_tokens(self) -> Text:
        text = Text("Context ", style="dim")
        used = self.context_tokens
        window = self.context_window
        text.append(f"{used:,}" if used is not None else "—")
        if window:
            text.append(f" / {window:,}")
        elif used is not None:
            text.append(" tok")
        session = f"{self.session_tokens:,} tok" if self.session_tokens is not None else "—"
        text.append(f" · session {session}")
        return text
