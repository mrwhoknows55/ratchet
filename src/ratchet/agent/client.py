import os

import httpx

from ratchet.agent.config import load_config
from ratchet.agent.responses import normalize_message

_transport: httpx.BaseTransport | None = None
_context_windows: dict[tuple[str, str], int] = {}


def _context_window(client: httpx.Client, model_cfg: dict, model_name: str,
                    api_key: str) -> int | None:
    configured = model_cfg.get("context_window")
    if type(configured) is int and configured > 0:
        return configured
    local = client.base_url.host in {"localhost", "127.0.0.1", "::1"}
    cache_key = (str(client.base_url), model_name)
    if not local and cache_key in _context_windows:
        return _context_windows[cache_key]
    url = client.base_url.copy_with(path="/api/v1/models") if local else "models"
    try:
        response = client.get(
            url, timeout=5 if not local else 2,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        if not local:
            for model in data.get("data", []):
                if not isinstance(model, dict) or model.get("id") != model_name:
                    continue
                provider = model.get("top_provider") or {}
                limits = [model.get("context_length")]
                if isinstance(provider, dict):
                    limits.append(provider.get("context_length"))
                limits = [limit for limit in limits if type(limit) is int and limit > 0]
                if limits:
                    _context_windows[cache_key] = min(limits)
                    return min(limits)
            return None
        for model in data.get("models", []):
            instances = model.get("loaded_instances", [])
            for instance in instances:
                if instance.get("id") == model_name or (
                    model.get("key") == model_name and len(instances) == 1
                ):
                    window = instance.get("config", {}).get("context_length")
                    if type(window) is int and window > 0:
                        return window
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    return None


def _request(messages: list[dict], model_cfg: dict, tools: list[dict] | None) -> dict:
    base_url = model_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = model_cfg.get("api_key", "not-needed")
    model_name = model_cfg.get("name", "qwen/qwen3-4b-2507")
    timeout = model_cfg.get("timeout", 10)

    payload = {"model": model_name, "messages": messages}
    if tools is not None:
        payload["tools"] = tools

    try:
        client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=_transport,
        )
        with client:
            response = client.post(
                "chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            reported_model = data.get("model")
            result = {"model": reported_model if isinstance(reported_model, str) and reported_model
                      else model_name, **normalize_message(message, tools)}
            usage = data.get("usage")
            if usage:
                result["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                window = _context_window(client, model_cfg, result["model"], api_key)
                if window is not None:
                    result["context_window"] = window
            return result
    except httpx.ConnectError:
        error_msg = (
            f"[LM Studio Offline] Could not connect to local server at {base_url}. "
            "Please start LM Studio server."
        )
        return {"content": error_msg, "model": model_name, "status": "offline", "error": error_msg}
    except httpx.HTTPStatusError as e:
        error_msg = f"[API Error] {e}"
        return {"content": error_msg, "model": model_name, "status": "error", "error": str(e)}
    except Exception as e:
        error_msg = f"[Error] Unexpected error: {str(e)}"
        return {"content": error_msg, "model": model_name, "status": "error", "error": str(e)}


def _model_config(base: dict, overrides: dict) -> dict:
    model_cfg = dict(base)
    if overrides.get("name", model_cfg.get("name")) != model_cfg.get("name"):
        model_cfg.pop("context_window", None)
    model_cfg.update(overrides)
    return model_cfg


def call_llm(
    messages: list[dict], override_config: dict | None = None, tools: list[dict] | None = None
) -> dict:
    cfg = load_config()
    base = cfg.get("model", {})
    overrides = (override_config or {}).get("model", {})
    model_cfg = _model_config(base, overrides)
    env_name = os.environ.get("OPENAI_MODEL", model_cfg.get("name", "qwen/qwen3-4b-2507"))
    if env_name != model_cfg.get("name", "qwen/qwen3-4b-2507"):
        model_cfg.pop("context_window", None)
    model_cfg.update(
        name=env_name,
        base_url=os.environ.get(
            "OPENAI_BASE_URL", model_cfg.get("base_url", "http://localhost:1234/v1")
        ),
        api_key=os.environ.get("OPENAI_API_KEY", model_cfg.get("api_key", "not-needed")),
    )
    result = _request(messages, model_cfg, tools)
    if result["status"] == "success":
        return result
    fallback = next((model for model in cfg.get("models", {}).values()
                     if model.get("fallback") is True and model.get("name")
                     and model.get("base_url")
                     and (model["name"], model["base_url"].rstrip("/")) != (
                         model_cfg["name"], model_cfg["base_url"].rstrip("/")
                     )), None)
    if fallback is None:
        return result
    fallback_cfg = _model_config(base, fallback)
    fallback_cfg["api_key"] = fallback.get(
        "api_key", os.environ.get("OPENAI_API_KEY", base.get("api_key", "not-needed"))
    )
    recovered = _request(messages, fallback_cfg, tools)
    recovered["fallback_from"] = result["model"]
    if recovered["status"] != "success":
        recovered["content"] = f"{result['content']}\n[Fallback Error] {recovered['content']}"
    return recovered
