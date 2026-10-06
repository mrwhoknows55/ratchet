import os

import httpx

from ratchet.agent.config import load_config

_transport: httpx.BaseTransport | None = None


def _context_window(client: httpx.Client, model_cfg: dict, model_name: str,
                    api_key: str) -> int | None:
    configured = model_cfg.get("context_window")
    if type(configured) is int and configured > 0:
        return configured
    if client.base_url.host not in {"localhost", "127.0.0.1", "::1"}:
        return None
    try:
        response = client.get(
            client.base_url.copy_with(path="/api/v1/models"), timeout=2,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        for model in response.json().get("models", []):
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


def call_llm(
    messages: list[dict], override_config: dict | None = None, tools: list[dict] | None = None
) -> dict:
    cfg = load_config()
    model_cfg = dict(cfg.get("model", {}))
    if override_config:
        overrides = override_config.get("model", {})
        if overrides.get("name", model_cfg.get("name")) != model_cfg.get("name"):
            model_cfg.pop("context_window", None)
        model_cfg.update(overrides)

    base_url = os.environ.get("OPENAI_BASE_URL", model_cfg.get("base_url", "http://localhost:1234/v1"))
    api_key = os.environ.get("OPENAI_API_KEY", model_cfg.get("api_key", "not-needed"))
    model_name = os.environ.get("OPENAI_MODEL", model_cfg.get("name", "qwen/qwen3-4b-2507"))
    if model_name != model_cfg.get("name", "qwen/qwen3-4b-2507"):
        model_cfg.pop("context_window", None)
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
            content = message["content"] or ""
            result = {"content": content, "model": model_name, "status": "success"}
            tool_calls = message.get("tool_calls")
            if tool_calls:
                result["tool_calls"] = tool_calls
            usage = data.get("usage")
            if usage:
                result["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                window = _context_window(client, model_cfg, model_name, api_key)
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
