import os

import httpx

TAVILY_BASE_URL = "https://api.tavily.com"
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 20
DEFAULT_WEB_TIMEOUT = 30
MAX_CONTENT_CHARS = 20000

_transport: httpx.BaseTransport | None = None


def _error(message: str) -> dict[str, str | int]:
    return {"stdout": "", "stderr": message, "exit_code": 1}


def _post(endpoint: str, payload: dict) -> tuple[dict | None, dict[str, str | int] | None]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None, _error(
            "Error: TAVILY_API_KEY is not set, so web tools are unavailable. "
            "Use the local file tools instead."
        )
    try:
        client = httpx.Client(
            base_url=TAVILY_BASE_URL,
            timeout=DEFAULT_WEB_TIMEOUT,
            transport=_transport,
        )
        with client:
            response = client.post(
                endpoint, json=payload, headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return response.json(), None
    except httpx.ConnectError:
        return None, _error(f"Could not reach the Tavily API at {TAVILY_BASE_URL}.")
    except httpx.HTTPStatusError as e:
        return None, _error(f"Tavily API error: {e.response.status_code} for {endpoint}.")
    except httpx.TimeoutException:
        return None, _error(f"Tavily API timed out after {DEFAULT_WEB_TIMEOUT}s.")
    except Exception as e:
        return None, _error(f"Web request failed: {e}")


def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, str | int]:
    term = query.strip()
    if not term:
        return _error("Error: 'query' must not be empty.")

    payload = {
        "query": term,
        "max_results": max(1, min(max_results, MAX_RESULTS_LIMIT)),
        "search_depth": "basic",
    }
    data, error = _post("/search", payload)
    if error is not None:
        return error

    results = data.get("results") or []
    if not results:
        return {"stdout": f"No results for '{term}'", "stderr": "", "exit_code": 0}

    blocks = [
        "\n".join(
            [
                f"{index}. {item.get('title', '(untitled)')}",
                f"   {item.get('url', '')}",
                f"   {item.get('content', '')}",
            ]
        )
        for index, item in enumerate(results, 1)
    ]
    return {"stdout": "\n\n".join(blocks), "stderr": "", "exit_code": 0}


def fetch_url(url: str) -> dict[str, str | int]:
    target = url.strip()
    if not target.startswith(("http://", "https://")):
        return _error(f"Error: 'url' must start with http:// or https://: '{url}'")

    data, error = _post("/extract", {"urls": [target], "format": "markdown"})
    if error is not None:
        return error

    results = data.get("results") or []
    if not results:
        failed = data.get("failed_results") or []
        reason = failed[0].get("error", "unknown error") if failed else "no content returned"
        return _error(f"Could not extract '{target}': {reason}")

    content = results[0].get("raw_content") or ""
    if len(content) > MAX_CONTENT_CHARS:
        content = (
            content[:MAX_CONTENT_CHARS]
            + f"\n[truncated: showing {MAX_CONTENT_CHARS} of {len(content)} characters.]"
        )
    return {"stdout": content, "stderr": "", "exit_code": 0}
