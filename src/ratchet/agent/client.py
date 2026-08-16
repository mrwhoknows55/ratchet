import os

import httpx

from ratchet.agent.config import load_config

_transport: httpx.BaseTransport | None = None


def call_llm(messages: list[dict], override_config: dict | None = None) -> dict:
    cfg = load_config()
    model_cfg = dict(cfg.get("model", {}))
    if override_config:
        model_cfg.update(override_config.get("model", {}))

    base_url = os.environ.get("OPENAI_BASE_URL", model_cfg.get("base_url", "http://localhost:1234/v1"))
    api_key = os.environ.get("OPENAI_API_KEY", model_cfg.get("api_key", "not-needed"))
    model_name = os.environ.get("OPENAI_MODEL", model_cfg.get("name", "qwen/qwen3-4b-2507"))
    timeout = model_cfg.get("timeout", 10)

    try:
        client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=_transport,
        )
        with client:
            response = client.post(
                "chat/completions",
                json={"model": model_name, "messages": messages},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"] or ""
            return {"content": content, "model": model_name, "status": "success"}
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
