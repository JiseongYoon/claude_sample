"""ToolCallingModel adapter over llm_serving.

Bridges the orchestrator's wire-format-agnostic `ToolCallingModel` protocol to the real
`LLMServingModule.chat`, which speaks the OpenAI-compatible chat-completions dict. Parsing
lives here so the loop stays decoupled from the response shape and from `llm_serving`.

Robust by design: a missing/odd field yields a safe `AssistantTurn` (text, or a tool call
with empty args) rather than raising — the gate still classifies whatever tool name comes
through. Transport / engine-not-ready errors propagate so the loop ends the run as `error`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from .loop import AssistantTurn, ToolCall

logger = logging.getLogger(__name__)


class ChatModule(Protocol):
    """The slice of `LLMServingModule` the adapter needs."""

    async def chat(self, messages: list[dict], **params: Any) -> dict: ...


class StreamingChatModule(ChatModule, Protocol):
    """A `ChatModule` that also streams. `chat_stream` calls `on_token` with each
    text delta and returns the SAME full response dict `chat` returns (reassembled)."""

    async def chat_stream(self, messages: list[dict], on_token: Any, **params: Any) -> dict: ...


def _parse_args(raw: Any) -> dict:
    """Tool-call arguments are an OpenAI JSON string (sometimes already a dict). Anything
    that doesn't yield a dict → `{}` (the bare tool name is still classified by the gate)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_turn(response: dict) -> AssistantTurn:
    """Map an OpenAI chat-completions response dict to an `AssistantTurn`.

    Robustness contract: this NEVER raises, for ANY input. Every level is guarded with
    an explicit `isinstance` check (a wrong-typed field → safe empty/text turn rather
    than a crash), and an outer catch is the final backstop against any unforeseen shape
    — a malformed engine response must degrade to an observation, not crash the loop turn."""
    try:
        return _parse_turn(response)
    except Exception: # noqa: BLE001 — robustness backstop; never propagate a parse fault
        logger.warning("parse_turn fell through to backstop on response shape", exc_info=True)
        return AssistantTurn(text="")


def _parse_turn(response: dict) -> AssistantTurn:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        return AssistantTurn(text="")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return AssistantTurn(text="")
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list): # tool_calls absent or odd → treat as none
        raw_calls = []
    calls: list[ToolCall] = []
    for i, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict): # a truthy non-dict function → skip (no .get crash)
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        cid = call.get("id") or f"call-{i}"
        calls.append(ToolCall(id=str(cid), tool=name, args=_parse_args(fn.get("arguments"))))
    if calls:
        content = message.get("content")
        return AssistantTurn(text=content if isinstance(content, str) else None,
                             tool_calls=calls)
    content = message.get("content")
    return AssistantTurn(text=content if isinstance(content, str) else "")


def _openai_assistant(m: dict) -> dict:
    """Render the loop's internal assistant tool-call message to the OpenAI wire shape."""
    calls: list[dict] = []
    for c in m.get("tool_calls", []):
        if not isinstance(c, dict):
            continue
        args = c.get("args", {})
        calls.append({
            "id": str(c.get("id") or ""),
            "type": "function",
            "function": {"name": c.get("tool") or "",
                         "arguments": args if isinstance(args, str) else json.dumps(args)},
        })
    out: dict = {"role": "assistant", "content": m.get("content") or ""}
    if calls:
        out["tool_calls"] = calls
    return out


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Translate the orchestrator loop's INTERNAL message shape to the OpenAI chat wire shape.

    The loop (`loop.py`) records an assistant tool-call turn in its own compact form
    `{"role":"assistant","content":…,"tool_calls":[{"id","tool","args"}]}`. The OpenAI
    `/v1/chat/completions` API expects each tool_call as
    `{"id","type":"function","function":{"name","arguments":<JSON string>}}`. The adapter owns the
    *request* mapping here (it already owns the *response* mapping in `parse_turn`) so the loop
    stays wire-format-agnostic and the engine receives well-formed prior tool calls. Every other
    message (system / user / text-only assistant / `role:tool` results / client-supplied history)
    is already OpenAI-shaped and passes through unchanged. Never raises (non-dict items skipped)."""
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            out.append(_openai_assistant(m))
        else:
            out.append(m)
    return out


class LLMToolModel:
    """`ToolCallingModel` over a `ChatModule`. `tools` schemas are passed straight through
    to `chat(..., tools=...)`; the engine returns tool calls the loop dispatches. The loop's
    internal message shape is translated to the OpenAI wire shape before each call."""

    def __init__(self, chat_module: ChatModule, **default_params: Any) -> None:
        self._chat = chat_module
        self._default_params = default_params

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        params = dict(self._default_params)
        if tools:
            params["tools"] = tools
        response = await self._chat.chat(to_openai_messages(messages), **params)
        return parse_turn(response)

    async def complete_stream(self, messages: list[dict], tools: list[dict],
                              on_token: Any) -> AssistantTurn:
        """Streaming variant: forwards text deltas to `on_token` as they arrive,
        but still ACCUMULATES the full turn (text + reassembled tool_calls) and parses it with the
        SAME `parse_turn` at end-of-stream — so the loop dispatches a complete, well-formed turn and
        tool-calling reliability is preserved (partial-argument fragments are never parsed)."""
        params = dict(self._default_params)
        if tools:
            params["tools"] = tools
        response = await self._chat.chat_stream(to_openai_messages(messages), on_token, **params)
        return parse_turn(response)
