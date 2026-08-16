import os
from pathlib import Path

import pytest

from ratchet.agent import config as agent_config

DEFAULT_MODEL_CONFIG = {
    "provider": "lmstudio",
    "base_url": "http://localhost:1234/v1",
    "name": "qwen/qwen3-4b-2507",
    "api_key": "not-needed",
    "timeout": 10,
}


@pytest.fixture(autouse=True)
def _isolate_config_path(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_config, "CONFIG_FILE", tmp_path / "config.toml")
    return tmp_path


def test_load_config_returns_defaults_when_file_missing():
    assert agent_config.load_config() == {"model": DEFAULT_MODEL_CONFIG}


def test_load_config_reads_existing_toml_file():
    agent_config.CONFIG_FILE.write_text(
        """
        [model]
        provider = "lmstudio"
        base_url = "http://localhost:1234/v1"
        name = "liquid/lfm2.5-1.2b"
        api_key = "not-needed"
        timeout = 20
        """
    )

    cfg = agent_config.load_config()

    assert cfg["model"]["name"] == "liquid/lfm2.5-1.2b"
    assert cfg["model"]["timeout"] == 20


def test_load_config_returns_defaults_on_malformed_toml():
    agent_config.CONFIG_FILE.write_text("this is not [valid toml")

    assert agent_config.load_config() == {"model": DEFAULT_MODEL_CONFIG}


def test_load_config_returns_defaults_on_empty_file():
    agent_config.CONFIG_FILE.write_text("")

    assert agent_config.load_config() == {"model": DEFAULT_MODEL_CONFIG}


def test_load_config_path_is_project_root_config_toml():
    assert agent_config.CONFIG_FILE.name == "config.toml"
    assert isinstance(agent_config.CONFIG_FILE, Path)


def test_load_config_loads_api_key_from_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-or-test-123\n")
    monkeypatch.setattr(agent_config, "ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent_config.load_config()

    assert os.environ.get("OPENAI_API_KEY") == "sk-or-test-123"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_load_config_does_not_override_existing_env_var(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-or-from-dotenv\n")
    monkeypatch.setattr(agent_config, "ENV_FILE", env_file)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-from-shell")

    agent_config.load_config()

    assert os.environ["OPENAI_API_KEY"] == "sk-or-from-shell"


def test_load_config_handles_missing_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_config, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert agent_config.load_config() == {"model": DEFAULT_MODEL_CONFIG}
