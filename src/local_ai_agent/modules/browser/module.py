"""BrowserModule — the registry-facing browser/web capability.

Builds the `GuardedFetcher` (over the concrete `HttpxFetcher` — URL/SSRF contained, IP-pinned) and,
if a SearXNG endpoint is configured, a `SearxngBackend`, then exposes the gated `Tool`s (`.tools`)
the composition root registers on the dispatcher (`web_search` safe-listed, `open_url`
gated via the existing `_NAV_TOOLS`). Reaching external hosts, it declares no in-registry dependency;
`health` reflects configuration (search backend present?) — reachability is handled gracefully at
tool-call time. The llm-driven synthesize tool (`web_answer`) is added in .
"""
from __future__ import annotations

from ...config import Settings
from ...core.module import Health, HealthStatus, ModuleSpec
from .config import BrowserConfig
from .fetcher import GuardedFetcher
from .tools import build_tools
from .transport import HttpxFetcher, SearxngBackend


class BrowserModule:
    """`Module` exposing the `browser` capability (web search + contained fetch + synthesize).

    `chat` (the llm-serving `ChatModel`) is optional: when present, the `web_answer` synthesize tool
    is added; otherwise only `web_search`/`open_url` are exposed. `depends_on=()` keeps search/fetch
    available even when llm-serving is down — `web_answer` degrades gracefully at call time.
    """

    def __init__(self, settings: Settings, chat: object | None = None, *,
                 fetcher: object | None = None, search: object | None = None) -> None:
        cfg = BrowserConfig.from_settings(settings)
        self._config = cfg
        # `fetcher`/`search` are a test-only injection seam: a fake `Fetcher`/`SearchBackend`
        # so the real module/tool wiring runs without real HTTP/SearXNG. None (production) → the real
        # IP-pinned `GuardedFetcher(HttpxFetcher())` + `SearxngBackend` (if a URL is configured).
        self._fetcher = fetcher if fetcher is not None else GuardedFetcher(
            HttpxFetcher(),
            max_bytes=cfg.max_bytes,
            per_fetch_timeout=cfg.per_fetch_timeout,
            max_redirects=cfg.max_redirects,
            allow_hosts=cfg.allow_hosts or None, # empty → blocklist mode
            name="browser",
        )
        if search is not None:
            self._search = search
        else:
            self._search = (
                SearxngBackend(cfg.searxng_url, timeout=cfg.per_fetch_timeout)
                if cfg.searxng_url else None
            )
        self._tools = build_tools(self._fetcher, self._search, config=cfg, chat=chat)
        self._started = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(
            name="browser", version="0.1.0",
            capabilities=("browser",), depends_on=(),
            description="web search + contained fetch (gated tools)",
        )

    @property
    def tools(self) -> list:
        return list(self._tools)

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        for closeable in (self._fetcher, self._search):
            if closeable is not None:
                try:
                    await closeable.close()
                except Exception: # noqa: BLE001 — best-effort teardown
                    pass
        self._started = False

    def health(self) -> Health:
        if not self._started:
            return Health(HealthStatus.absent, "not started")
        if self._search is None:
            return Health(HealthStatus.degraded, "no search backend configured (fetch only)")
        return Health(HealthStatus.ok, "ready")
