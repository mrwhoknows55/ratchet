import pytest

from ratchet.agent import config as agent_config
from ratchet.agent.models import load_supported_models


@pytest.fixture(autouse=True)
def _isolate_config_path(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_config, "CONFIG_FILE", tmp_path / "config.toml")
    return tmp_path


def test_load_supported_models_reads_models_table_from_config_toml():
    agent_config.CONFIG_FILE.write_text(
        """
        [model]
        provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        name = "anthropic/claude-sonnet-5"
        api_key = "not-needed"
        timeout = 20

        [models.claude]
        name = "anthropic/claude-sonnet-5"

        [models.lmstudio]
        name = "liquid/lfm2.5-1.2b"
        base_url = "http://localhost:1234/v1"
        """
    )

    assert load_supported_models() == {
        "claude": {"name": "anthropic/claude-sonnet-5"},
        "lmstudio": {
            "name": "liquid/lfm2.5-1.2b",
            "base_url": "http://localhost:1234/v1",
        },
    }


def test_load_supported_models_returns_empty_dict_when_no_models_table():
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

    assert load_supported_models() == {}
