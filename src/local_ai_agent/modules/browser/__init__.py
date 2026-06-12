"""Browser / web-access capability.

Web search → fetch+extract → synthesize-with-citations, driven by the local model, behind a
pluggable fetcher/search seam and gated on the dispatcher. v1 is *fetch-parse-first*
(httpx + readability/trafilatura + SearXNG); a JS-rendering browser (Playwright/browser-use as a
passive driver) is deferred to sub-phase 6.1 behind out-of-process egress isolation.

 ships the seam itself — the `Fetcher` / `SearchBackend` Protocols (raw ops, implemented by
concrete fetchers in later steps), the `GuardedFetcher` policy wrapper (URL/SSRF containment + the
per-hop redirect re-check + size cap), the typed `BrowserError` hierarchy, `contain_url` (the SSRF
crux), and the browser config model — all network-free and testable with an injected fake fetcher
and a fake resolver.
"""
