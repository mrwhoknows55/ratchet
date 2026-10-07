import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from ratchet.agent.plan import create_plan, get_plan
from ratchet.agent.tools import SYSTEM_PROMPT


@dataclass
class Session:
    path: Path
    id: str
    messages: list[dict]
    usage: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.path.parent / "sessions" / self.id

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "session.json").write_text(json.dumps({
            "messages": self.messages, "usage": self.usage,
        }), encoding="utf-8")
        self.path.write_text(json.dumps({"session_id": self.id}), encoding="utf-8")

    @classmethod
    def switch(cls, path: Path, session_id: str) -> str:
        normalized = str(UUID(session_id.strip()))
        if not (path.parent / "sessions" / normalized / "session.json").is_file():
            raise FileNotFoundError(f"unknown session: {normalized}")
        path.write_text(json.dumps({"session_id": normalized}), encoding="utf-8")
        return normalized

    @classmethod
    def new(cls, path: Path) -> "Session":
        session = cls(path, str(uuid4()), [{"role": "system", "content": SYSTEM_PROMPT}])
        session.save()
        return session

    @classmethod
    def open(cls, path: Path, workspace: Path) -> "Session":
        legacy = None
        try:
            pointer = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(pointer, list):
                legacy = pointer
            elif isinstance(pointer, dict):
                session_id = str(UUID(pointer["session_id"]))
                root = path.parent / "sessions" / session_id
                data = json.loads((root / "session.json").read_text(encoding="utf-8"))
                messages = data["messages"]
                if isinstance(messages, list) and isinstance(data.get("usage", {}), dict):
                    return cls(path, session_id, messages, data.get("usage", {}))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        session = cls.new(path)
        if legacy:
            session.messages = legacy
            result = get_plan(workspace)
            if result["exit_code"] == 0:
                plan = json.loads(result["stdout"])["plan"]
                create_plan(session.root, plan["title"], plan["steps"],
                            plan["summary"], plan["risks"])
            session.save()
        return session
