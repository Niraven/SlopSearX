"""Tests for FastAPI server — /search and /health endpoints."""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

import engines  # noqa: F401 — triggers @register_engine
from slopsearx.adapter import (
    AdapterResponse,
    EngineAdapter,
    EngineStatus,
    SearchResult,
    register_engine,
)
from slopsearx.config import load_config
from slopsearx.server import _check_cache, app

# ---------------------------------------------------------------------------
# Test engine — mock adapter for controlled test scenarios
# ---------------------------------------------------------------------------


@register_engine
class _MockEngine(EngineAdapter):
    """Mock engine used only in tests — not registered during normal startup."""

    name = "mocktest"
    display_name = "Mock Test Engine"
    env_prefix = "ENGINE_MOCKTEST"
    engine_type = "api"
    categories = ["general", "news", "tech", "science"]

    async def search(self, query, params=None):
        if query == "error":
            return AdapterResponse(
                results=[],
                status=EngineStatus.ERROR,
                error_message="simulated error",
            )
        if query == "timeout_sim":
            return AdapterResponse(
                results=[],
                status=EngineStatus.TIMEOUT,
                error_message="simulated timeout",
            )
        if query == "blocked":
            return AdapterResponse(
                results=[],
                status=EngineStatus.BLOCKED,
                error_message="CAPTCHA detected",
            )
        if query == "rate_limited":
            return AdapterResponse(
                results=[],
                status=EngineStatus.RATE_LIMITED,
                error_message="too many requests",
            )
        if query == "precondition":
            return AdapterResponse(
                results=[],
                status=EngineStatus.ERROR,
                error_message="missing route-specific credential",
                circuit_breaker_failure=False,
            )
        if query == "leak_exception":
            # Raise an exception with an embedded URL to test server-level sanitization
            raise RuntimeError(
                "Client error '403 Forbidden' for url 'https://api.example.com/search?key=secret-key-12345&q=test'"
            )

        # Normal response
        return AdapterResponse(
            results=[
                SearchResult(
                    url=f"https://mock{i}.com",
                    title=f"Mock Result {i}",
                    content=f"Content for mock result {i}.",
                    engine=self.name,
                )
                for i in range(3)
            ],
            status=EngineStatus.OK,
            latency_ms=42.0,
        )


class _EmptyScrapeEngine(EngineAdapter):
    """Successful scrape response with no parsed results."""

    name = "emptyscrape"
    engine_type = "scrape"
    categories = ["general"]

    async def search(self, query, params=None):
        return AdapterResponse(results=[], status=EngineStatus.OK)


class _RouteEngine(EngineAdapter):
    """Engine double whose label proves which effective route dispatched."""

    name = "route"
    categories = ["general"]

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self.calls = 0
        self.last_categories: list[str] = []

    async def search(self, query, params=None):
        self.calls += 1
        self.last_categories = list((params or {}).get("categories", []))
        return AdapterResponse(
            results=[
                SearchResult(
                    url=f"https://example.com/{self.label}",
                    title=self.label,
                    content=self.label,
                    engine=self.label,
                )
            ],
            status=EngineStatus.OK,
        )


class _MemoryCache:
    """Small in-memory cache double for route-isolation tests."""

    is_connected = True

    def __init__(self) -> None:
        self.search: dict[str, dict] = {}
        self.answers: dict[str, dict] = {}

    async def get(self, key: str):
        return self.search.get(key)

    async def set(self, key: str, value: dict, _ttl: int):
        self.search[key] = value

    async def get_answer(self, query: str):
        raise AssertionError("query-only answer cache must not be read")

    async def set_answer(self, query: str, value: dict):
        raise AssertionError("query-only answer cache must not be written")


# ---------------------------------------------------------------------------
# Test fixture: server with mock engines enabled
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Test client with mock engine as the only active engine.

    We modify the server's _active_engines directly to control
    what engines are available during tests.
    """
    import slopsearx.server as server_mod

    # Save original state
    original_engines = dict(server_mod._active_engines)
    original_empty_scrape_diagnostics = server_mod._empty_scrape_diagnostics_enabled

    with TestClient(app) as tc:
        # Set mock engine AFTER startup runs (which calls discover_engines)
        server_mod._active_engines = {
            "mocktest": _MockEngine(),
        }
        # Disable query router so mock engines aren't filtered to Tier 1
        server_mod._router = None
        yield tc

    # Restore original state
    server_mod._active_engines = original_engines
    server_mod._empty_scrape_diagnostics_enabled = original_empty_scrape_diagnostics


# ---------------------------------------------------------------------------
# Cache hit handling
# ---------------------------------------------------------------------------


class TestCheckCache:
    """Shared cache-hit response handling."""

    async def test_returns_none_when_cache_is_disconnected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import slopsearx.server as server_mod

        cache = type("Cache", (), {"is_connected": False})()
        monkeypatch.setattr(server_mod, "_cache", cache)

        response = await _check_cache(cache, lambda _: _unexpected_cache_read(), "ssx-test")

        assert response is None

    async def test_returns_none_on_cache_miss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import slopsearx.server as server_mod

        cache = type("Cache", (), {"is_connected": True})()
        monkeypatch.setattr(server_mod, "_cache", cache)

        response = await _check_cache(cache, lambda _: _cache_result(None), "ssx-test")

        assert response is None

    async def test_marks_successful_cache_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import slopsearx.server as server_mod

        cache = type("Cache", (), {"is_connected": True})()
        monkeypatch.setattr(server_mod, "_cache", cache)
        cached: dict[str, object] = {"meta": {"cached": False}, "results": []}

        response = await _check_cache(cache, lambda _: _cache_result(cached), "ssx-test")

        assert response is not None
        assert response.status_code == 200
        assert cached == {"meta": {"cached": True}, "results": []}

    async def test_returns_cached_error_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import slopsearx.server as server_mod

        cache = type("Cache", (), {"is_connected": True})()
        monkeypatch.setattr(server_mod, "_cache", cache)

        response = await _check_cache(cache, lambda _: _cache_result({"_error": True}), "ssx-test")

        assert response is not None
        assert response.status_code == 503
        assert response.body == (
            b'{"error":"service_unavailable","message":"Temporarily unavailable '
            b'(cached error)","meta":{"cached":true,"query_id":"ssx-test"}}'
        )


async def _cache_result(value: dict[str, object] | None) -> dict[str, object] | None:
    return value


async def _unexpected_cache_read() -> None:
    raise AssertionError("disconnected cache must not be read")


# ---------------------------------------------------------------------------
# /search endpoint
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    """GET /search endpoint."""

    def test_basic_search(self, client: TestClient) -> None:
        """Basic search returns JSON with results."""
        response = client.get("/search", params={"q": "test query"})

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["number_of_results"] == 3
        assert len(data["results"]) == 3
        assert "meta" in data

    def test_missing_query(self, client: TestClient) -> None:
        """Missing q parameter returns 400."""
        response = client.get("/search")

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "query_required"

    def test_empty_query(self, client: TestClient) -> None:
        """Empty q parameter returns 400."""
        response = client.get("/search", params={"q": ""})

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "query_required"

    def test_whitespace_only_query(self, client: TestClient) -> None:
        """Whitespace-only query returns 400."""
        response = client.get("/search", params={"q": "   "})

        assert response.status_code == 400

    def test_yaml_format(self, client: TestClient) -> None:
        """format=yaml returns YAML+Markdown response."""
        response = client.get("/search", params={"q": "test", "format": "yaml"})

        assert response.status_code == 200
        assert "text/vnd.yaml+markdown" in response.headers["content-type"]
        assert "test" in response.text
        assert "## Results Summary" in response.text

    def test_format_is_canonicalized_before_cache_lookup(self, client: TestClient, monkeypatch) -> None:
        import slopsearx.server as server_mod

        monkeypatch.setattr(server_mod, "_cache", _MemoryCache())

        uppercase = client.get("/search", params={"q": "same format query", "format": "YAML"})
        lowercase = client.get("/search", params={"q": "same format query", "format": "yaml"})

        assert uppercase.status_code == 200
        assert lowercase.status_code == 200
        assert "text/vnd.yaml+markdown" in uppercase.headers["content-type"]
        assert "text/vnd.yaml+markdown" in lowercase.headers["content-type"]

    def test_unknown_format_is_rejected(self, client: TestClient) -> None:
        response = client.get("/search", params={"q": "test", "format": "html"})

        assert response.status_code == 400
        assert response.json()["error"] == "invalid_format"

    def test_cached_json_never_leaks_into_yaml_response(self, client: TestClient, monkeypatch) -> None:
        import slopsearx.server as server_mod

        monkeypatch.setattr(server_mod, "_cache", _MemoryCache())

        json_response = client.get("/search", params={"q": "same cached query"})
        yaml_response = client.get("/search", params={"q": "same cached query", "format": "yaml"})

        assert json_response.status_code == 200
        assert "text/vnd.yaml+markdown" in yaml_response.headers["content-type"]

    def test_cache_isolated_by_category(self, client: TestClient, monkeypatch) -> None:
        import slopsearx.server as server_mod

        cache = _MemoryCache()
        monkeypatch.setattr(server_mod, "_cache", cache)

        science = client.get("/search", params={"q": "same scoped query", "categories": "science"})
        news = client.get("/search", params={"q": "same scoped query", "categories": "news"})

        assert science.status_code == 200
        assert news.status_code == 200
        assert science.json()["meta"]["cached"] is False
        assert news.json()["meta"]["cached"] is False
        assert len(cache.search) == 2
        assert cache.answers == {}

    def test_cache_isolated_when_effective_engine_route_changes(
        self,
        client: TestClient,
        monkeypatch,
    ) -> None:
        import slopsearx.server as server_mod

        cache = _MemoryCache()
        first_engine = _RouteEngine("first")
        second_engine = _RouteEngine("second")
        monkeypatch.setattr(server_mod, "_cache", cache)

        server_mod._active_engines = {"first": first_engine}
        first = client.get("/search", params={"q": "same route query"})
        server_mod._active_engines = {"second": second_engine}
        second = client.get("/search", params={"q": "same route query"})

        assert first.json()["results"][0]["title"] == "first"
        assert second.json()["results"][0]["title"] == "second"
        assert first_engine.calls == 1
        assert second_engine.calls == 1
        assert len(cache.search) == 2

    def test_categories_are_canonicalized_before_dispatch(self, client: TestClient) -> None:
        import slopsearx.server as server_mod

        engine = _RouteEngine("category")
        server_mod._active_engines = {"route": engine}

        response = client.get(
            "/search",
            params={"q": "category query", "categories": " Science,REFERENCE ", "engines": "route"},
        )

        assert response.status_code == 200
        assert engine.last_categories == ["science", "reference"]

    def test_blocked_engine_eventually_opens_circuit(self, client: TestClient) -> None:
        import slopsearx.server as server_mod

        engine = _MockEngine()
        engine._circuit_threshold = 1
        server_mod._active_engines = {"mocktest": engine}

        blocked = client.get("/search", params={"q": "blocked"})
        after_block = client.get("/search", params={"q": "fresh query"})

        assert blocked.status_code == 503
        assert after_block.status_code == 503
        assert after_block.json()["unresponsive_engines"] == [["mocktest", "circuit open"]]

    def test_rate_limit_does_not_open_shared_circuit(self, client: TestClient) -> None:
        import slopsearx.server as server_mod

        engine = _MockEngine()
        engine._circuit_threshold = 1
        server_mod._active_engines = {"mocktest": engine}

        limited = client.get("/search", params={"q": "rate_limited"})
        unrelated = client.get("/search", params={"q": "fresh query"})

        assert limited.status_code == 503
        assert unrelated.status_code == 200
        assert engine.circuit_allowed()

    def test_route_precondition_error_does_not_open_shared_circuit(self, client: TestClient) -> None:
        import slopsearx.server as server_mod

        engine = _MockEngine()
        engine._circuit_threshold = 1
        server_mod._active_engines = {"mocktest": engine}

        precondition = client.get("/search", params={"q": "precondition"})
        unrelated = client.get("/search", params={"q": "fresh query"})

        assert precondition.status_code == 503
        assert unrelated.status_code == 200
        assert engine.circuit_allowed()

    def test_json_format_default(self, client: TestClient) -> None:
        """format=json is the default."""
        response = client.get("/search", params={"q": "test"})

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    def test_unresponsive_engine(self, client: TestClient) -> None:
        """Error from engine is reported in unresponsive_engines."""
        response = client.get("/search", params={"q": "error"})

        assert response.status_code == 503  # all engines unresponsive
        data = response.json()
        assert len(data["unresponsive_engines"]) == 1
        assert data["unresponsive_engines"][0][0] == "mocktest"

    def test_suggestions_always_present(self, client: TestClient) -> None:
        """suggestions field is always present (may be empty)."""
        response = client.get("/search", params={"q": "test"})

        data = response.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_meta_fields(self, client: TestClient) -> None:
        """meta.* extension fields are present."""
        response = client.get("/search", params={"q": "test"})

        data = response.json()
        meta = data["meta"]
        assert "response_time_ms" in meta
        assert "cached" in meta
        assert "query_id" in meta
        assert "engine_status" in meta
        assert meta["query_id"].startswith("ssx-")
        assert isinstance(meta["cached"], bool)

    def test_empty_scrape_diagnostic_is_opt_in(self, client: TestClient) -> None:
        """An empty scrape is visible without being marked unresponsive."""
        import slopsearx.server as server_mod

        server_mod._active_engines = {"emptyscrape": _EmptyScrapeEngine()}
        server_mod._empty_scrape_diagnostics_enabled = False

        disabled_response = client.get("/search", params={"q": "diagnostic-disabled"})

        assert "empty_engines" not in disabled_response.json()["meta"]

        server_mod._empty_scrape_diagnostics_enabled = True

        response = client.get("/search", params={"q": "diagnostic-enabled"})

        assert response.status_code == 200
        data = response.json()
        assert data["unresponsive_engines"] == []
        assert data["meta"]["empty_engines"] == [["emptyscrape", "successful scrape returned no results"]]

    def test_engines_filter(self, client: TestClient) -> None:
        """engines parameter filters which engines to use."""
        response = client.get("/search", params={"q": "test", "engines": "mocktest"})

        assert response.status_code == 200
        data = response.json()
        assert data["number_of_results"] == 3

    def test_nonexistent_engine_filter(self, client: TestClient) -> None:
        """Filtering to a nonexistent engine returns 503."""
        response = client.get("/search", params={"q": "test", "engines": "nonexistent"})

        assert response.status_code == 503

    def test_dispatch_engine_sanitizes_error_message(self, client: TestClient) -> None:
        """VAL-M1-013: _dispatch_engine broad except handler sanitizes error messages.

        When an adapter raises an exception with a URL containing an API key,
        the server-level handler must sanitize it before returning to the client.
        """
        response = client.get("/search", params={"q": "leak_exception"})

        assert response.status_code == 503  # all engines unresponsive
        data = response.json()
        assert len(data["unresponsive_engines"]) == 1
        error_msg = data["unresponsive_engines"][0][1]
        assert "secret-key-12345" not in error_msg, f"API key found in unresponsive_engines error: {error_msg}"

    def test_query_params_preserved(self, client: TestClient) -> None:
        """Query parameters are accepted without error."""
        response = client.get(
            "/search",
            params={
                "q": "test",
                "language": "fr",
                "pageno": 2,
                "safesearch": 1,
                "time_range": "month",
                "categories": "news,tech",
            },
        )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /health endpoint."""

    def test_health_ok(self, client: TestClient) -> None:
        """Health check returns status with per-engine info."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "engines" in data
        assert "mocktest" in data["engines"]
        # Mock engine health uses search("healthcheck") which returns
        # normal results, so status should be OK
        assert data["engines"]["mocktest"]["status"] == "ok"

    def test_health_no_engines(self) -> None:
        """Health works even with no engines registered."""
        import slopsearx.server as server_mod

        original = dict(server_mod._active_engines)
        server_mod._active_engines = {}

        try:
            with TestClient(app) as client:
                server_mod._active_engines = {}
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"
                assert data["engines"] == {}
        finally:
            server_mod._active_engines = original


class TestEngineConfigPropagation:
    """Ensures engine config from env vars reaches adapters."""

    def test_env_var_api_key_flows_to_adapter(self, monkeypatch) -> None:
        """ENGINE_BRAVE_API_KEY env var should reach Brave adapter's config."""
        monkeypatch.setenv("ENGINE_BRAVE_API_KEY", "test-key-12345")

        # Re-discover engines with env var set
        cfg = load_config()
        engine_configs = {name: dataclasses.asdict(entry) for name, entry in cfg.engines.items()}

        # Brave config should have the API key
        assert "brave" in engine_configs
        assert engine_configs["brave"]["api_key"] == "test-key-12345"
