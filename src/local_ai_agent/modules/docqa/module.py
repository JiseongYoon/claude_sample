"""DocQAModule — the registry-facing capability module.

A `core.module.Module` that advertises the `docqa` capability and `depends_on=("llm-serving",)`,
so the registry gates the capability off automatically whenever the serving dependency is
down/absent (fault isolation, ). It constructs the tools (`.tools`) which the
composition root registers on the dispatcher. The module owns no request
handling itself — the tools do, through the gated dispatcher.
"""
from __future__ import annotations

from ...config import Settings
from ...core.module import Health, HealthStatus, ModuleSpec
from .summarizer import ChatModel
from .tools import DocQAConfig, build_tools


class DocQAModule:
    """`Module` exposing the `docqa` capability (document QA & summarization over local docs)."""

    def __init__(self, settings: Settings, chat: ChatModel) -> None:
        self._config = DocQAConfig.from_settings(settings) # raises if docs_root unset
        self._tools = build_tools(self._config, chat)
        self._started = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(
            name="docqa", version="0.1.0",
            capabilities=("docqa",), depends_on=("llm-serving",),
            description="local document QA & summarization (gated tools)",
        )

    @property
    def tools(self) -> list:
        """The gated `Tool`s the composition root registers on the dispatcher."""
        return list(self._tools)

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def health(self) -> Health:
        # The module itself is just ready/not — the registry gates the capability on the
        # `llm-serving` dependency's health; the tools also degrade gracefully at runtime.
        if not self._started:
            return Health(HealthStatus.absent, "not started")
        return Health(HealthStatus.ok, "ready")
