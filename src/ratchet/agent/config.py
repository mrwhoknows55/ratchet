import tomllib
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent.parent.parent / "config.toml"

DEFAULT_CONFIG = {
    "model": {
        "provider": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "name": "qwen/qwen3-4b-2507",
        "api_key": "not-needed",
        "timeout": 10,
    }
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            data = tomllib.loads(CONFIG_FILE.read_text())
            if data:
                return data
        except tomllib.TOMLDecodeError:
            pass
    return DEFAULT_CONFIG
