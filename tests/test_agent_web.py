import httpx
import pytest

from ratchet.agent import web as agent_web


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setattr(agent_web, "_transport", None)


def _mock(handler):
    return httpx.MockTransport(handler)


def test_search_web_formats_title_url_and_snippet(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Messi", "url": "https://a.test", "content": "a footballer"}
                ]
            },
        )

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    result = agent_web.search_web("who is messi")

    assert result["exit_code"] == 0
    assert "Messi" in result["stdout"]
    assert "https://a.test" in result["stdout"]
    assert "a footballer" in result["stdout"]


def test_search_web_sends_query_and_bearer_key(monkeypatch):
    seen = {}

    def handler(request):
        import json

        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    agent_web.search_web("who is messi", max_results=3)

    assert seen["path"] == "/search"
    assert seen["auth"] == "Bearer tvly-test-key"
    assert seen["body"]["query"] == "who is messi"
    assert seen["body"]["max_results"] == 3


def test_search_web_clamps_max_results(monkeypatch):
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    agent_web.search_web("q", max_results=500)

    assert seen["body"]["max_results"] == agent_web.MAX_RESULTS_LIMIT


def test_search_web_reports_no_results(monkeypatch):
    monkeypatch.setattr(
        agent_web, "_transport", _mock(lambda request: httpx.Response(200, json={"results": []}))
    )
    result = agent_web.search_web("q")
    assert result["exit_code"] == 0
    assert "No results" in result["stdout"]


def test_search_web_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = agent_web.search_web("q")
    assert result["exit_code"] == 1
    assert "TAVILY_API_KEY" in result["stderr"]


def test_search_web_rejects_an_empty_query(monkeypatch):
    result = agent_web.search_web("   ")
    assert result["exit_code"] == 1
    assert "empty" in result["stderr"].lower()


def test_search_web_reports_an_http_error(monkeypatch):
    monkeypatch.setattr(
        agent_web, "_transport", _mock(lambda request: httpx.Response(401, json={"detail": "bad"}))
    )
    result = agent_web.search_web("q")
    assert result["exit_code"] == 1
    assert "401" in result["stderr"]


def test_search_web_reports_a_connection_failure(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no network", request=request)

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    result = agent_web.search_web("q")
    assert result["exit_code"] == 1
    assert "reach" in result["stderr"].lower()


def test_fetch_url_returns_page_content(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [{"url": "https://a.test", "raw_content": "# Title\n\nbody"}],
                "failed_results": [],
            },
        )

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    result = agent_web.fetch_url("https://a.test")

    assert result["exit_code"] == 0
    assert "# Title" in result["stdout"]
    assert "body" in result["stdout"]


def test_fetch_url_requests_markdown_from_the_extract_endpoint(monkeypatch):
    seen = {}

    def handler(request):
        import json

        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"raw_content": "x"}], "failed_results": []})

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    agent_web.fetch_url("https://a.test")

    assert seen["path"] == "/extract"
    assert seen["body"]["urls"] == ["https://a.test"]
    assert seen["body"]["format"] == "markdown"


def test_fetch_url_rejects_a_non_http_url(monkeypatch):
    result = agent_web.fetch_url("file:///etc/passwd")
    assert result["exit_code"] == 1
    assert "http" in result["stderr"].lower()


def test_fetch_url_reports_a_failed_extraction(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={"results": [], "failed_results": [{"url": "https://a.test", "error": "404"}]},
        )

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    result = agent_web.fetch_url("https://a.test")

    assert result["exit_code"] == 1
    assert "https://a.test" in result["stderr"]


def test_fetch_url_truncates_a_very_long_page(monkeypatch):
    body = "x" * (agent_web.MAX_CONTENT_CHARS + 500)

    def handler(request):
        return httpx.Response(
            200, json={"results": [{"raw_content": body}], "failed_results": []}
        )

    monkeypatch.setattr(agent_web, "_transport", _mock(handler))
    result = agent_web.fetch_url("https://a.test")

    assert "truncated" in result["stdout"]
    assert len(result["stdout"]) < len(body)


def test_fetch_url_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = agent_web.fetch_url("https://a.test")
    assert result["exit_code"] == 1
    assert "TAVILY_API_KEY" in result["stderr"]
