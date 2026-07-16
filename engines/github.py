"""GitHub API adapter — code, repository, and issue/PR search."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from slopsearx.adapter import (
    AdapterResponse,
    EngineAdapter,
    EngineStatus,
    SearchResult,
    register_engine,
)


@register_engine
class GitHubAdapter(EngineAdapter):
    name = "github"
    display_name = "GitHub"
    env_prefix = "ENGINE_GITHUB"
    engine_type = "api"
    categories = ["reference", "github:code", "github:issues", "github:prs"]

    def __init__(self, config: dict[str, Any] | None = None, rate_limiter: Any = None) -> None:
        super().__init__(config, rate_limiter)
        self._quota_lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def search(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> AdapterResponse:
        cfg = self.config
        token = cfg.get("api_key") or ""
        base_url = cfg.get("base_url", "https://api.github.com")
        timeout_ms = cfg.get("timeout_ms", 5_000)
        max_results = cfg.get("max_results", 5)
        categories = {
            str(category).strip().lower()
            for category in ((params or {}).get("categories", []) or ["general"])
            if str(category).strip()
        }

        # Determine sub-mode from categories
        if "github:code" in categories:
            endpoint = f"{base_url}/search/code"
        elif "github:issues" in categories or "github:prs" in categories:
            endpoint = f"{base_url}/search/issues"
        else:
            endpoint = f"{base_url}/search/repositories"

        if "/search/code" in endpoint and not token:
            return AdapterResponse(
                results=[],
                status=EngineStatus.ERROR,
                error_message="GitHub code search requires ENGINE_GITHUB_API_KEY",
                circuit_breaker_failure=False,
            )

        if early := await self._check_rate_limit():
            return early
        if quota_response := await self._check_provider_quota():
            return quota_response

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "SlopSearX/0.2.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        params_dict: dict[str, Any] = {
            "q": query,
            "per_page": max_results,
            "page": 1,
        }

        start_time = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout_ms / 1000.0) as client:
                resp = await client.get(endpoint, headers=headers, params=params_dict)
                latency = (time.monotonic() - start_time) * 1000

                rate_limit_remaining = resp.headers.get("x-ratelimit-remaining")
                if resp.status_code == 429 or (
                    resp.status_code == 403
                    and ("rate limit" in (resp.text or "").lower() or rate_limit_remaining == "0")
                ):
                    await self._extend_provider_cooldown(resp.headers)
                    return AdapterResponse(
                        results=[],
                        status=EngineStatus.RATE_LIMITED,
                        error_message="rate limited by GitHub",
                        latency_ms=latency,
                    )
                if resp.status_code == 401:
                    return AdapterResponse(
                        results=[],
                        status=EngineStatus.ERROR,
                        error_message="GitHub authentication rejected",
                        latency_ms=latency,
                        circuit_breaker_failure=False,
                    )
                if resp.status_code == 403:
                    return AdapterResponse(results=[], status=EngineStatus.BLOCKED, latency_ms=latency)
                if resp.status_code == 422:
                    # Code search needs more specific qualifiers; return empty gracefully
                    return AdapterResponse(results=[], status=EngineStatus.OK, latency_ms=latency)
                resp.raise_for_status()

                data = resp.json()
                items = data.get("items", [])
                results = self._parse_items(items, query, endpoint)
                return AdapterResponse(results=results, status=EngineStatus.OK, latency_ms=latency)

        except httpx.TimeoutException:
            latency = (time.monotonic() - start_time) * 1000
            return AdapterResponse(results=[], status=EngineStatus.TIMEOUT, latency_ms=latency)
        except Exception as exc:  # noqa: BLE001
            latency = (time.monotonic() - start_time) * 1000
            return AdapterResponse(
                results=[],
                status=EngineStatus.ERROR,
                error_message=str(exc),
                latency_ms=latency,
            )

    async def _check_provider_quota(self) -> AdapterResponse | None:
        """Enforce the conservative GitHub search quota before egress."""

        try:
            rate = float(self.config.get("rate_limit", 0.15))
        except (TypeError, ValueError):
            rate = 0.15
        if rate <= 0:
            return AdapterResponse(
                results=[],
                status=EngineStatus.RATE_LIMITED,
                error_message="GitHub provider quota disabled",
                circuit_breaker_failure=False,
            )

        async with self._quota_lock:
            now = time.monotonic()
            if now < self._next_request_at:
                return AdapterResponse(
                    results=[],
                    status=EngineStatus.RATE_LIMITED,
                    error_message="GitHub provider quota guard",
                    circuit_breaker_failure=False,
                )
            self._next_request_at = now + (1.0 / rate)
        return None

    async def _extend_provider_cooldown(self, headers: httpx.Headers) -> None:
        """Honor GitHub's explicit retry window without sleeping a request."""

        delays: list[float] = []
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                delays.append(max(0.0, float(retry_after)))
            except ValueError:
                pass

        reset_at = headers.get("x-ratelimit-reset")
        if reset_at:
            try:
                delays.append(max(0.0, float(reset_at) - time.time()))
            except ValueError:
                pass

        if not delays:
            return

        async with self._quota_lock:
            retry_at = time.monotonic() + max(delays)
            self._next_request_at = max(self._next_request_at, retry_at)

    def _parse_items(self, items: list[dict[str, Any]], query: str, endpoint: str) -> list[SearchResult]:
        """Parse GitHub API search results into SearchResult list."""
        results: list[SearchResult] = []

        is_code = "/search/code" in endpoint
        is_issues = "/search/issues" in endpoint

        for idx, item in enumerate(items):
            if is_code:
                # Code search results
                repo_name = (item.get("repository") or {}).get("full_name", "")
                path = item.get("path", "")
                url = f"https://github.com/{repo_name}/blob/main/{path}" if repo_name else item.get("html_url", "")
                title = f"{repo_name}: {path}" if repo_name else path
                content = item.get("text_matches", [{}])[0].get("fragment", "") if item.get("text_matches") else ""
                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        content=content[:300] if content else f"Code file in {repo_name}",
                        engine=self.name,
                        position=idx + 1,
                        category="code",
                    ),
                )
            elif is_issues:
                # Issue/PR search results
                url = item.get("html_url", "")
                title = item.get("title", "")
                state = item.get("state", "")
                raw_labels = item.get("labels") or []
                labels = ", ".join(lbl["name"] for lbl in raw_labels if isinstance(lbl, dict))
                content = item.get("body", "") or ""
                clean = content.strip()[:300] if content else ""
                if labels:
                    clean = f"[{state}] [{labels}] {clean}" if clean else f"[{state}] [{labels}]"
                else:
                    clean = f"[{state}] {clean}" if clean else f"[{state}]"
                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        content=clean.strip(),
                        engine=self.name,
                        position=idx + 1,
                        category="issues",
                        published_date=item.get("created_at", ""),
                    ),
                )
            else:
                # Repository search results
                url = item.get("html_url", "")
                title = item.get("full_name", item.get("name", ""))
                desc = item.get("description") or ""
                lang = item.get("language") or ""
                stars = item.get("stargazers_count", 0)
                topics = ", ".join(item.get("topics", []) or [])
                detail_parts = [f"★ {stars}"]
                if lang:
                    detail_parts.append(lang)
                if topics:
                    detail_parts.append(topics)
                content = f"{desc} — {' | '.join(detail_parts)}" if desc else " — ".join(detail_parts)
                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        content=content,
                        engine=self.name,
                        position=idx + 1,
                        score=float(stars),
                        category="repositories",
                    ),
                )

        return results
