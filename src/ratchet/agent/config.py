import tomllib
from pathlib import Path

from dotenv import load_dotenv

CONFIG_FILE = Path(__file__).parent.parent.parent.parent / "config.toml"
ENV_FILE = Path(__file__).parent.parent.parent.parent / ".env"

DEFAULT_CONFIG = {
    "model": {
        "provider": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "name": "qwen/qwen3-4b-2507",
        "api_key": "not-needed",
        "timeout": 10,
    },
    "agent": {
        "max_steps": 12,
        "command_timeout": 10,
    },
    "subagent": {
        "max_steps": 8,
        "max_depth": 1,
    },
}


def load_config() -> dict:
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
    if CONFIG_FILE.exists():
        try:
            data = tomllib.loads(CONFIG_FILE.read_text())
            if data:
                return data
        except tomllib.TOMLDecodeError:
            pass
    return DEFAULT_CONFIG
