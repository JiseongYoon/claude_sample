"""LLM-Serving — OpenAI-compatible client wrapped as a Module.

Separate from the model-manager (per the requirements architecture). Provides the
`chat` capability by talking to whatever engine the model-manager has running.

Loose coupling: it depends on a structural `ServingTarget` (the model-manager
satisfies it) injected by the composition root — it never imports the
model-manager module. Its `health()` reflects serve-readiness (`target.is_serving`)
so `chat` is gated off whenever no model is loaded, on top of the registry's
`depends_on` module gating.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Protocol, runtime_checkable

from ..config import Settings
from ..core.module import Health, HealthStatus, ModuleSpec


@runtime_checkable
class ServingTarget(Protocol):
    """What llm-serving needs from the engine owner (the model-manager)."""

    @property
    def is_serving(self) -> bool: ...
    @property
    def base_url(self) -> str: ...


class EngineNotReady(RuntimeError):
    """Raised when a serving call is attempted with no model ready."""


# transport: (base_url, payload) -> parsed response dict
ChatTransport = Callable[[str, dict], Awaitable[dict]]
# token callback: receives each text delta as it streams.
OnToken = Callable[[str], Awaitable[None]]
# stream transport: (base_url, payload, on_token, read_timeout) -> the REASSEMBLED full response dict
StreamTransport = Callable[[str, dict, OnToken, float], Awaitable[dict]]


async def _httpx_chat_transport(base_url: str, payload: dict) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()


class _StreamAccum:
    """Accumulate-then-parse: fold OpenAI SSE chat-completion *chunks* into one
    full non-streaming response dict, so the existing `parse_turn` works unchanged at end-of-stream.
    Text deltas are surfaced (for live display) via `add()`'s return; tool-call fragments are
    reassembled by `index` — name (first non-empty) + arguments (concatenated string fragments) —
    so partial-JSON argument fragments are NEVER parsed mid-stream (that would break tool-calling).
    Pure + hermetically testable (no httpx). Never raises on an odd chunk shape."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._tool_calls: dict[int, dict] = {}

    def add(self, chunk: dict) -> str | None:
        """Fold one SSE chunk; return its text delta (if any) for the on_token callback."""
        if not isinstance(chunk, dict):
            return None
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return None
        text = delta.get("content")
        has_text = isinstance(text, str) and text != ""
        if has_text:
            self._content.append(text)
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                if not isinstance(idx, int):
                    idx = 0
                slot = self._tool_calls.setdefault(idx, {"id": None, "name": None, "args": []})
                if isinstance(tc.get("id"), str) and tc["id"]:
                    slot["id"] = tc["id"]
                fn = tc.get("function")
                if isinstance(fn, dict):
                    if isinstance(fn.get("name"), str) and fn["name"]:
                        slot["name"] = fn["name"]
                    arg = fn.get("arguments")
                    if isinstance(arg, str):
                        slot["args"].append(arg)
        return text if has_text else None

    def response(self) -> dict:
        """The reassembled full OpenAI response dict (same shape the buffered transport returns)."""
        content = "".join(self._content)
        calls: list[dict] = []
        for idx in sorted(self._tool_calls):
            slot = self._tool_calls[idx]
            if not slot["name"]:
                continue
            calls.append({"id": slot["id"] or f"call-{idx}", "type": "function",
                          "function": {"name": slot["name"], "arguments": "".join(slot["args"])}})
        msg: dict = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = calls
        return {"choices": [{"message": msg}]}


def _parse_sse_line(line: str) -> dict | None:
    """Parse one SSE line into a chat-completion chunk dict, or None for a blank / non-`data:` /
    `[DONE]` / malformed line (all skipped). Pure + testable (no I/O); never raises."""
    import json as _json

    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if not data or data == "[DONE]":
        return None
    try:
        chunk = _json.loads(data)
    except ValueError:
        return None
    return chunk if isinstance(chunk, dict) else None


async def _httpx_stream_transport(base_url: str, payload: dict, on_token: OnToken,
                                  read_timeout: float) -> dict:
    """Consume llama-server's OpenAI SSE stream, surfacing text deltas to `on_token` and returning
    the reassembled full response dict. `read_timeout` bounds the gap BETWEEN tokens (a stalled
    engine raises `httpx.ReadTimeout` rather than hanging). `async with client.stream` closes the
    connection on cancel; CancelledError propagates to the caller's cleanup."""
    import httpx

    accum = _StreamAccum()
    timeout = httpx.Timeout(read_timeout, connect=10.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                chunk = _parse_sse_line(line)
                if chunk is None:
                    continue
                delta = accum.add(chunk)
                if delta:
                    await on_token(delta)
    return accum.response()


_PARAM_KEYS = ("temperature", "top_p", "top_k", "max_tokens")


class LLMServingModule:
    """`Module` exposing the `chat` capability over the running engine."""

    def __init__(self, settings: Settings, target: ServingTarget,
                 transport: ChatTransport | None = None,
                 stream_transport: StreamTransport | None = None) -> None:
        self._settings = settings
        self._target = target
        self._transport = transport or _httpx_chat_transport
        self._stream_transport = stream_transport or _httpx_stream_transport
        self._started = False
        self._default_params: dict = {} # request-level defaults set via the param API

    @property
    def spec(self) -> ModuleSpec:
        # advertise `multimodal` ONLY when a vision projector is configured (operator
        # prerequisite). Combined with the existing health-gating (serving must be up), the registry
        # then reports `multimodal` AVAILABLE only when serving + a projector is present — the single
        # truth the image-ingest allowlist and the multimodal DocQA tool gate on.
        caps = ("chat", "multimodal") if self._settings.model_mmproj_file else ("chat",)
        return ModuleSpec(
            name="llm-serving", version="0.1.0",
            capabilities=caps, depends_on=("model-manager",),
            description="OpenAI-compatible chat over the loaded model",
        )

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def health(self) -> Health:
        if not self._started:
            return Health(HealthStatus.absent, "not started")
        if self._target.is_serving:
            return Health(HealthStatus.ok, "serving")
        # not a fault — there is simply no model ready to serve yet → unavailable
        return Health(HealthStatus.absent, "no model ready")

    # -- request-level default params (set via the param API, ) --------
    def get_default_params(self) -> dict:
        return dict(self._default_params)

    def set_default_params(self, params: dict) -> None:
        """Merge in known request-level params (unknown keys ignored)."""
        self._default_params.update({k: v for k, v in params.items()
                                     if k in _PARAM_KEYS and v is not None})

    def _build_payload(self, messages: list[dict], params: dict) -> dict:
        """Assemble the chat-completions request body shared by `chat` + `chat_stream`."""
        payload: dict = {"model": self._settings.served_model_name, "messages": messages}
        merged = dict(self._default_params)
        for k in _PARAM_KEYS:
            if params.get(k) is not None:
                merged[k] = params[k]
        for k in _PARAM_KEYS:
            if merged.get(k) is not None:
                payload[k] = merged[k]
        # (F1): the agent path (LLMToolModel) passes tool schemas via `tools=`.
        # `tools` is NOT a `_PARAM_KEYS` member, so it was silently dropped here — the engine never
        # saw the tool definitions and could never propose a tool call. Forward it explicitly when
        # present (a non-empty list); direct-chat callers pass no tools and are unaffected.
        tools = params.get("tools")
        if isinstance(tools, list) and tools:
            payload["tools"] = tools
        return payload

    async def chat(self, messages: list[dict], **params) -> dict:
        """Send a chat-completion request to the running engine.

        Per-call params override the stored defaults. Raises `EngineNotReady` if
        no model is loaded/ready (defense-in-depth; the gateway also gates)."""
        if not self._target.is_serving:
            raise EngineNotReady("no model is loaded/ready")
        return await self._transport(self._target.base_url, self._build_payload(messages, params))

    async def chat_stream(self, messages: list[dict], on_token: OnToken, **params) -> dict:
        """Streaming chat: sets `stream:true` and consumes the SSE stream, calling
        `on_token(text_delta)` as tokens arrive and returning the SAME reassembled full response
        dict the buffered `chat` returns (so `parse_turn` is unchanged). `stream` is set EXPLICITLY
        here (it is not a `_PARAM_KEYS` member, so it can't ride through `**params`). The buffered
        `chat` path is untouched."""
        if not self._target.is_serving:
            raise EngineNotReady("no model is loaded/ready")
        payload = self._build_payload(messages, params)
        payload["stream"] = True
        return await self._stream_transport(self._target.base_url, payload, on_token,
                                            self._settings.llm_stream_idle_timeout_s)
