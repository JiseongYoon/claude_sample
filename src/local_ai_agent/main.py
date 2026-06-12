"""Composition root — the SINGLE place modules are constructed and wired.

This is the one entry point that knows about concrete modules; everything else
depends only on the `Module` seam. Capability modules (phase 4+) get added to the
list in `build_application` and nowhere else — that is what keeps the system
loosely coupled and fault-isolated.

For there are no capability modules yet, so `build_application` wires an
empty `Application`. Step-2 added the module registry (health/gating); 
hands the composed app to the FastAPI gateway and serves it with uvicorn here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import FastAPI

from .config import Settings, get_settings
from .core.app import Application
from .core.gateway import create_gateway
from .core.module import Module


@dataclass(frozen=True)
class BuildOverrides:
    """Optional test-only fakes for the leaf dependencies that touch external resources
    (GPU / Docker / SSH / network). **Every field defaults to None → the real leaf is built and
    production behaviour is unchanged.** This is the injection seam the composed integration
    test uses to drive the REAL composition root hermetically (faking only the leaves), so the test
    exercises the actual wiring rather than a parallel hand-composed harness.

    - `serving_module`: a fake Module that satisfies ChatModel (`chat(...)`) AND reports `Health.ok`
      with `spec.name == "llm-serving"` — replaces the model stack so DocQA's `depends_on=("llm-serving",)`
      is satisfied without a live `llama-server`.
    - `agent_model`: a `ToolCallingModel` for the `AgentRuntime` (else `LLMToolModel(serving)`).
    - `storage_connectors`: a `{name: GuardedConnector}` map (else loaded from `storage_connectors_file`).
    - `browser_fetcher` / `browser_search`: a `Fetcher` / `SearchBackend` (else `HttpxFetcher`/`SearxngBackend`).
    - `exec_runner`: a `ProcRunner` (else the real docker CLI runner).
    - `mcp_client_factory`: a `ClientFactory` (else the real `SdkMcpClient`).
    - `audit_sink`: an `AuditSink` for the `AgentRuntime` (else `LoggingAuditSink`) — lets a test assert
      audit completeness (INV-6) over the composed graph.
    """

    serving_module: object | None = None
    agent_model: object | None = None
    storage_connectors: object | None = None
    browser_fetcher: object | None = None
    browser_search: object | None = None
    exec_runner: object | None = None
    mcp_client_factory: object | None = None
    audit_sink: object | None = None


def build_application(settings: Settings | None = None, *,
                      overrides: BuildOverrides | None = None) -> Application:
    """Build and wire the application graph (the composition root).

    `settings` defaults to the process config (`get_settings()`); pass an explicit
    one in tests. Add capability modules to `modules` here as phases land — this
    function is the only wiring site. `overrides` (test-only) injects fake leaf
    dependencies so the real wiring can be exercised hermetically; it is None in
    production (see `BuildOverrides`).
    """
    settings = settings or get_settings()
    ov = overrides or BuildOverrides()
    modules: list[Module] = [
        # phase 2+: ModelManager(settings), DocQA(...), Browser(...), ...
    ]
    if settings.enable_demo_modules:
        # fault-isolation demo (off in production by default)
        from .modules.demo import DependentModule, EchoModule, FlakyModule

        modules += [
            EchoModule(),
            FlakyModule(broken=settings.demo_flaky_broken),
            DependentModule(),
        ]
    serving = None
    if ov.serving_module is not None:
        # test seam: a fake serving Module (ChatModel + Health.ok) stands in for the model stack so
        # capability health-gating (DocQA depends_on llm-serving) holds without a live llama-server.
        serving = ov.serving_module
        modules.append(serving)
    elif (settings.enable_model_stack or settings.enable_agent or settings.enable_docqa
            or settings.enable_browser):
        # model stack: manager (process control) + serving client (depends on it).
        # The agent loop, docqa, and browser's `web_answer` all need a serving client (ChatModel),
        # so any of those flags builds the stack.
        from .modules.llm_serving import LLMServingModule
        from .modules.model_manager.module import ModelManagerModule

        manager = ModelManagerModule(settings)
        serving = LLMServingModule(settings, manager)
        modules += [manager, serving]

    # the DocQA capability module (gated on llm-serving health).
    docqa_module = None
    if settings.enable_docqa:
        from .modules.docqa.module import DocQAModule

        docqa_module = DocQAModule(settings, chat=serving) # serving satisfies ChatModel
        modules.append(docqa_module)

    # the storage capability module (remote file access; no in-registry dep).
    storage_module = None
    if settings.enable_storage:
        from .modules.storage.module import StorageModule

        # when docqa is also on, pass the serving ChatModel so the module also exposes the
        # remote → DocQA tools (`summarize_remote`/`answer_remote`); otherwise just the 6 file tools.
        storage_chat = serving if settings.enable_docqa else None
        storage_module = StorageModule(settings, chat=storage_chat,
                                       connectors=ov.storage_connectors) # fail-fast on bad config
        modules.append(storage_module)

    # the browser capability module (web search + contained fetch; URL/SSRF guarded).
    browser_module = None
    if settings.enable_browser:
        from .modules.browser.module import BrowserModule

        # serving (built above when enable_browser) satisfies ChatModel → enables `web_answer`;
        # depends_on=() keeps search/fetch up even if llm-serving is down (web_answer degrades).
        browser_module = BrowserModule(settings, chat=serving,
                                       fetcher=ov.browser_fetcher, search=ov.browser_search)
        modules.append(browser_module)

    # the exec capability module (sandboxed command/code execution; no in-registry dep).
    exec_module = None
    if settings.enable_exec:
        from .modules.exec.module import ExecModule

        # exec needs no llm-serving; it talks to the Docker daemon. Fail-closed at start() if unreachable.
        exec_module = ExecModule(settings, runner=ov.exec_runner)
        modules.append(exec_module)

    # the MCP capability module (external servers' tools as gated agent tools).
    mcp_module = None
    if settings.enable_mcp:
        from .modules.mcp.module import McpModule

        # MCP talks to external (untrusted) servers, no in-registry dep. Tool names are DISCOVERED at
        # connect, so the module self-registers its tools on the dispatcher in start() (see bind_register
        # below) rather than via the static-tools loop. Best-effort: a down server degrades only itself.
        mcp_module = McpModule(settings, client_factory=ov.mcp_client_factory)
        modules.append(mcp_module)

    # presentation/avatar hostability demo (§A5). Proves the GENERIC Module interface +
    # the existing REST/WS API host a future presentation/Live2D module with ZERO core change — this
    # standard one-flag append is the ENTIRE wiring (no gateway route / registry / gate change).
    if settings.enable_presentation_demo:
        from .modules.presentation.module import PresentationModule

        modules.append(PresentationModule())

    application = Application(modules=modules)

    # the gated file-ingestion store (upload → `_ingest/` sandbox under docs_root,
    # referenced by opaque id). Built only when DocQA + docs_root are present, since an ingested
    # file is read back through DocQA's `docs_root` sandbox. DF1 persistent — rebuild ids from disk.
    application.ingest_store = None
    if settings.enable_docqa and settings.docs_root is not None:
        from .modules.docqa import loaders
        from .modules.docqa.ingest import IngestPolicy, IngestStore

        # when a vision projector is configured (multimodal), the ingest type allowlist also
        # admits raster image types; with text-only serving they are NOT accepted (the endpoint 415s).
        allowed_exts = set(loaders._SUPPORTED)
        if settings.model_mmproj_file:
            from .modules.docqa.render import IMAGE_EXTS

            allowed_exts |= IMAGE_EXTS
        ingest_policy = IngestPolicy(
            settings.docs_root,
            max_file_bytes=settings.ingest_max_file_bytes,
            max_files=settings.ingest_max_files,
            max_total_bytes=settings.ingest_max_total_bytes,
            allowed_extensions=frozenset(allowed_exts),
        )
        ingest_store = IngestStore(ingest_policy)
        ingest_store.rebuild_from_disk() # ids survive a restart (no-op if `_ingest/` absent)
        application.ingest_store = ingest_store

    # The proven safety chain (gate → approvals → dispatcher) is the single gated path to
    # tools. Build it when the agent loop, docqa, or storage needs it; the agent runtime + WS
    # surface are built only for `enable_agent`.
    if (settings.enable_agent or settings.enable_docqa or settings.enable_storage
            or settings.enable_browser or settings.enable_exec or settings.enable_mcp):
        from .modules.safety.approval import PendingApprovals
        from .modules.safety.dispatcher import ToolDispatcher
        from .modules.safety.gate import DEFAULT_SAFE_TOOLS, SafetyGate

        # read-only capability tools are safe to allowlist; the gate's path rules
        # (nonlocal/secret/system) still force confirm/blocked on dangerous path args, and
        # mutation tools keep their gate-classified `needs_confirmation`.
        safe_tools = set(DEFAULT_SAFE_TOOLS)
        if settings.enable_docqa:
            from .modules.docqa.tools import SAFE_TOOL_NAMES

            safe_tools |= SAFE_TOOL_NAMES
            # the multimodal vision tool is safe-listed (a read) ONLY when a projector is
            # configured AND an ingest store exists (the visual path is otherwise absent).
            if settings.model_mmproj_file and application.ingest_store is not None:
                from .modules.docqa.multimodal import MULTIMODAL_SAFE_TOOL_NAMES

                safe_tools |= MULTIMODAL_SAFE_TOOL_NAMES
        if settings.enable_storage:
            from .modules.storage.tools import STORAGE_SAFE_TOOL_NAMES

            safe_tools |= STORAGE_SAFE_TOOL_NAMES
            if settings.enable_docqa: # remote → DocQA read tools are also safe-listed
                from .modules.storage.remote_docqa import REMOTE_DOCQA_SAFE_TOOL_NAMES

                safe_tools |= REMOTE_DOCQA_SAFE_TOOL_NAMES
        if settings.enable_browser:
            from .modules.browser.tools import BROWSER_SAFE_TOOL_NAMES

            safe_tools |= BROWSER_SAFE_TOOL_NAMES # `web_search` only; `open_url` stays gated via _NAV_TOOLS
        if settings.enable_exec:
            from .modules.exec.tools import EXEC_SAFE_TOOL_NAMES

            safe_tools |= EXEC_SAFE_TOOL_NAMES # `read_workspace_file` only; run/write stay gated
        if settings.enable_mcp:
            from .modules.mcp.tools import MCP_SAFE_TOOL_NAMES

            safe_tools |= MCP_SAFE_TOOL_NAMES # EMPTY — external MCP tools are NEVER safe-listed

        gate = SafetyGate(safe_tools=frozenset(safe_tools))
        approvals = PendingApprovals(timeout_seconds=settings.approval_timeout_seconds)
        dispatcher = ToolDispatcher(gate=gate, approvals=approvals, tools=[])
        for cap_module in (docqa_module, storage_module, browser_module, exec_module):
            if cap_module is not None:
                for tool in cap_module.tools: # real capability tools on the dispatcher
                    dispatcher.register(tool)
        # register the gated multimodal tool ONLY when a projector is configured + an ingest
        # store exists — so the visual path is reachable (via the gate, INV-1) only when the model can serve it.
        if settings.enable_docqa and settings.model_mmproj_file and application.ingest_store is not None:
            from .modules.docqa.multimodal import AnswerAboutImageTool

            dispatcher.register(AnswerAboutImageTool(
                application.ingest_store, serving,
                max_pages=settings.multimodal_max_pdf_pages,
                max_page_px=settings.multimodal_max_page_px,
                max_render_bytes=settings.multimodal_max_render_bytes,
                render_timeout_s=settings.multimodal_render_timeout_s,
                answer_max_tokens=settings.docqa_answer_max_tokens,
            ))
        # MCP tools are discovered at connect (dynamic), so the module self-registers them on the
        # dispatcher during start() via this injected callback (the dispatcher exists only now).
        if mcp_module is not None:
            mcp_module.bind_register(dispatcher.register)

        if settings.enable_agent:
            # compose the safety chain into a runnable agent bundle.
            from .modules.orchestrator.model_adapter import LLMToolModel
            from .modules.orchestrator.runtime import AgentRuntime
            from .modules.safety.audit import LoggingAuditSink

            # test seam: an injected ToolCallingModel drives the loop deterministically; else the real
            # adapter over the serving ChatModel.
            agent_model = ov.agent_model if ov.agent_model is not None else LLMToolModel(serving)
            audit_sink = ov.audit_sink if ov.audit_sink is not None else LoggingAuditSink()
            application.agent_runtime = AgentRuntime(
                gate=gate,
                approvals=approvals,
                dispatcher=dispatcher,
                model=agent_model,
                audit=audit_sink,
                max_steps=settings.agent_max_steps,
                max_tool_calls=settings.agent_max_tool_calls,
                result_char_cap=settings.agent_result_char_cap,
                deadline_seconds=settings.agent_deadline_seconds,
                decision_timeout=settings.approval_decision_timeout_seconds,
                max_history_messages=settings.agent_max_history_messages, # history_char_cap=settings.agent_history_char_cap, # ingest_store=application.ingest_store, # attachment resolution
            )
    return application


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the composed application and wrap it in the API gateway.

    This is the single object graph the server (and tests) use: composition root
    → registry-backed Application → secure FastAPI gateway. The gateway's lifespan
    drives `Application.startup/shutdown`.
    """
    settings = settings or get_settings()
    return create_gateway(build_application(settings), settings)


def main() -> None:
    """Process entry point (console script `local-ai-agent`) — serve the API."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    # the gateway binds `api_port` (DISTINCT from `serve_port`, the model server port) so the
    # model-manager's `llama-server --port serve_port` never collides with the gateway.
    uvicorn.run(create_app(settings), host=settings.serve_host, port=settings.api_port)


if __name__ == "__main__":
    main()
