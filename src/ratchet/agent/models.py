from ratchet.agent.config import load_config


def load_supported_models() -> dict[str, dict[str, str]]:
    return load_config().get("models", {})
