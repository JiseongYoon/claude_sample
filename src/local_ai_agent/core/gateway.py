"""API gateway — the FastAPI app that exposes the core over REST + WebSocket.

The gateway owns no capability logic; it is a thin, secure adapter over the
`Application`/`ModuleRegistry`. Capability availability (registry, ) is
surfaced here and *gated* at the API boundary: an unavailable capability yields a
clean 503, an unknown one a 404 — never a 500.

Security baseline (auth tokens arrive in ):
- TrustedHost allowlist (anti host-spoofing),
- CORS deny-by-default (explicit allowlist, never wildcard),
- security headers, generic error handler (no internal leakage),
- health-detail redaction (don't leak exception text/paths to clients).
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request

from ..config import Settings, get_settings
from .app import Application
from .auth import Principal, authenticate_ws, create_access_token, make_auth_dependencies
from .module import Health, HealthStatus

logger = logging.getLogger(__name__)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


# map the ingest core's typed errors → HTTP status + a FIXED generic detail (the
# class names are duck-typed so the gateway keeps its no-core→modules-import discipline). The
# detail never echoes the attacker filename / a path / a stack — only a category.
_INGEST_ERROR_STATUS = {
    "TooLarge": 413, "QuotaExceeded": 413, "UnsupportedType": 415,
    "BadFilename": 400, "TooMany": 429, "ContainmentError": 409,
}
_INGEST_STATUS_DETAIL = {
    413: "file too large", 415: "unsupported file type", 400: "invalid file",
    429: "too many files", 409: "conflict",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach conservative security headers to every response."""

    async def dispatch(self, request: Request, call_next): # noqa: ANN001
        response = await call_next(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response


def _overall_status(snap: dict[str, Health]) -> HealthStatus:
    """Coarse rollup: down if any module down, else degraded if any degraded,
    else ok (ok also for an empty system — the platform itself is up)."""
    statuses = {h.status for h in snap.values()}
    if HealthStatus.down in statuses:
        return HealthStatus.down
    if HealthStatus.degraded in statuses:
        return HealthStatus.degraded
    return HealthStatus.ok


def _module_view(snap: dict[str, Health], *, expose_detail: bool) -> dict:
    """Public module view — statuses only; detail included only if allowed."""
    if expose_detail:
        return {n: {"status": h.status.value, "detail": h.detail} for n, h in snap.items()}
    return {n: h.status.value for n, h in snap.items()}


class LoadRequest(BaseModel):
    gguf_file: str | None = None


class ParamsRequest(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)


class ChatRequest(BaseModel):
    messages: list[dict] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    # optional ingested-doc ids; the toolless direct path loads their text (bounded)
    # into a system context message (no tool loop, no gate — same posture as direct chat today).
    attachments: list[str] | None = Field(default=None, max_length=16)


def create_gateway(app: Application, settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI gateway over a composed `Application`."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await app.startup()
        try:
            yield
        finally:
            await app.shutdown()

    api = FastAPI(title="Local AI Agent — Core API", version="0.1.0", lifespan=lifespan)

    # --- security middleware (outermost first) ---
    api.add_middleware(SecurityHeadersMiddleware)
    if settings.trusted_hosts:
        api.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.trusted_hosts))
    if settings.cors_allow_origins: # explicit allowlist only; never wildcard
        api.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            # `X-API-Key` is one of the API's own auth headers — a browser client uses it to bootstrap a
            # token (POST /auth/token) and for api-key-scheme REST calls. Omitting it here makes the CORS
            # preflight reject those requests, so the Web UI can't mint/authenticate cross-origin.
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    # --- auth dependencies (bound to this settings instance) ---
    require_principal, require_scope, require_api_key = make_auth_dependencies(settings)
    if settings.auth_enabled and not settings.api_key and not settings.jwt_secret:
        logger.warning(
            "auth enabled but no API key / JWT secret configured — "
            "protected routes will reject all requests (set API_KEY / JWT_SECRET in .env)"
        )
    if settings.jwt_secret and len(settings.jwt_secret) < 32:
        logger.warning(
            "JWT_SECRET is shorter than 32 bytes — use a longer random secret for HS256 "
            "(e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`)"
        )

    # --- generic error handler: no internal leakage ---
    @api.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception): # noqa: ANN202
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # --- auth: issue a scoped JWT (bootstrap-authenticated by the API key) ---
    @api.post("/auth/token")
    async def issue_token(body: dict | None = None, _: Principal = Depends(require_api_key)):
        body = body or {}
        scopes = body.get("scopes") or ["read", "invoke"]
        subject = body.get("subject") or "client"
        expires = body.get("expires_minutes")
        token = create_access_token(settings, subject, scopes, expires_minutes=expires)
        ttl_min = expires if expires is not None else settings.jwt_expiry_minutes
        return {"access_token": token, "token_type": "bearer",
                "expires_in": ttl_min * 60, "scopes": scopes}

    # --- REST (health is public; the rest require authentication) ---
    @api.get("/health")
    async def health():
        snap = app.health_snapshot()
        status = _overall_status(snap)
        body = {"status": status.value,
                "modules": _module_view(snap, expose_detail=settings.expose_health_detail)}
        code = 503 if status is HealthStatus.down else 200
        return JSONResponse(status_code=code, content=body)

    @api.get("/capabilities")
    async def capabilities(_: Principal = Depends(require_principal)):
        available = sorted(app.available_capabilities())
        all_caps = sorted({c for m in app.registry.modules for c in m.spec.capabilities})
        return {"available": available, "all": all_caps}

    @api.get("/modules/{name}")
    async def module_status(name: str, _: Principal = Depends(require_principal)):
        if app.get_module(name) is None:
            raise HTTPException(status_code=404, detail="unknown module")
        h = app.registry.health(name)
        view = {"name": name, "status": h.status.value, "available": app.registry.is_module_available(name)}
        if settings.expose_health_detail:
            view["detail"] = h.detail
        return view

    @api.get("/capabilities/{name}")
    async def capability_status(name: str, _: Principal = Depends(require_principal)):
        return {"name": name, "available": app.is_capability_available(name)}

    # --- read-only discovery endpoints: list-only, NO authority, no secret/path
    # leak. Each module owns its non-secret projection; the gateway only serializes it. A
    # missing module degrades to an empty list (not an error) so the UI renders gracefully. ---
    @api.get("/storage/connectors")
    async def storage_connectors(_: Principal = Depends(require_principal)):
        mod = app.get_module("storage")
        if mod is None or not hasattr(mod, "connector_views"):
            return {"connectors": [], "health": None}
        return {"connectors": mod.connector_views(),
                "health": app.registry.health("storage").status.value}

    @api.get("/mcp/servers")
    async def mcp_servers(_: Principal = Depends(require_principal)):
        mod = app.get_module("mcp")
        if mod is None or not hasattr(mod, "server_views"):
            return {"servers": [], "health": None}
        return {"servers": mod.server_views(),
                "health": app.registry.health("mcp").status.value}

    @api.get("/tools")
    async def list_tools(_: Principal = Depends(require_principal)):
        runtime = getattr(app, "agent_runtime", None)
        if runtime is None:
            return {"tools": []}
        # capability per tool — built from the modules that own tools (.tools); read-only.
        cap_of: dict[str, str] = {}
        for m in app.registry.modules:
            mod_tools = getattr(m, "tools", None)
            if not mod_tools:
                continue
            cap = m.spec.capabilities[0] if m.spec.capabilities else None
            for t in mod_tools:
                nm = getattr(t, "name", None)
                if isinstance(nm, str):
                    cap_of[nm] = cap
        roster = runtime.dispatcher.roster()
        for r in roster:
            r["capability"] = cap_of.get(r["name"])
        return {"tools": roster}

    @api.post("/capabilities/{name}/invoke")
    async def invoke_capability(name: str, payload: dict | None = None,
                                _: Principal = Depends(require_scope("invoke"))):
        # gate at the API boundary: unknown -> 404, unavailable -> 503
        known = any(name in m.spec.capabilities for m in app.registry.modules)
        if not known:
            raise HTTPException(status_code=404, detail="unknown capability")
        if not app.is_capability_available(name):
            raise HTTPException(status_code=503, detail="capability unavailable")
        return {"capability": name, "status": "ok", "echo": payload or {}} # stub until real caps

    # --- model stack: modules accessed structurally, no concrete import ---
    def _require_module(name: str):
        mod = app.get_module(name)
        if mod is None:
            raise HTTPException(status_code=503, detail="model stack not enabled")
        return mod

    @api.get("/model/status")
    async def model_status(_: Principal = Depends(require_principal)):
        mgr = _require_module("model-manager")
        body = {"state": mgr.engine_state.value, "loaded_file": mgr.loaded_file,
                "serving": mgr.is_serving}
        if settings.expose_health_detail:
            body["detail"] = mgr.health().detail
        return body

    @api.get("/model/files")
    async def model_files(_: Principal = Depends(require_principal)):
        """List the GGUF files available to load — bare *.gguf names in MODEL_GGUF_DIR, sorted.
        Names only (no absolute paths); a glob never escapes the dir. 503 if the model stack is off."""
        _require_module("model-manager") # 503 when the model stack is disabled
        base = Path(settings.model_gguf_dir)
        try:
            names = sorted(p.name for p in base.glob("*.gguf") if p.is_file())
        except OSError:
            logger.warning("could not list model_gguf_dir", exc_info=True)
            names = []
        return {"files": names}

    @api.post("/model/load")
    async def model_load(body: LoadRequest, response: Response,
                         principal: Principal = Depends(require_scope("model:admin"))):
        mgr = _require_module("model-manager")
        try:
            state, launched = mgr.launch_load(body.gguf_file)
        except ValueError as exc:
            logger.warning("model load rejected (invalid file): %r", exc)
            raise HTTPException(status_code=400, detail="invalid gguf_file") from None
        logger.info("model load by %r: file=%r launched=%s", principal.subject, body.gguf_file, launched)
        response.status_code = 202 if launched else 200
        return {"state": state.value, "accepted": launched}

    @api.post("/model/switch")
    async def model_switch(body: LoadRequest, response: Response,
                           principal: Principal = Depends(require_scope("model:admin"))):
        mgr = _require_module("model-manager")
        if not body.gguf_file:
            raise HTTPException(status_code=400, detail="gguf_file is required")
        try:
            state, launched = mgr.launch_switch(body.gguf_file)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid gguf_file") from None
        except Exception as exc: # ModelBusy (duck-typed) → conflict
            if exc.__class__.__name__ == "ModelBusy":
                raise HTTPException(status_code=409, detail="model busy (load in progress)") from None
            raise
        logger.info("model switch by %r: file=%r", principal.subject, body.gguf_file)
        response.status_code = 202 if launched else 200
        return {"state": state.value, "accepted": launched}

    @api.post("/model/unload")
    async def model_unload(principal: Principal = Depends(require_scope("model:admin"))):
        mgr = _require_module("model-manager")
        state = await mgr.unload()
        logger.info("model unload by %r", principal.subject)
        return {"state": state.value}

    @api.get("/model/params")
    async def get_params(_: Principal = Depends(require_principal)):
        llm = _require_module("llm-serving")
        return {"params": llm.get_default_params()}

    @api.post("/model/params")
    async def set_params(body: ParamsRequest, _: Principal = Depends(require_scope("model:admin"))):
        mgr = _require_module("model-manager")
        llm = _require_module("llm-serving")
        if not mgr.is_serving:
            raise HTTPException(status_code=409, detail="no model loaded — cannot tune params")
        params = body.model_dump(exclude_none=True)
        if "max_tokens" in params:
            params["max_tokens"] = min(params["max_tokens"], settings.chat_max_tokens_ceiling)
        llm.set_default_params(params)
        return {"params": llm.get_default_params()}

    @api.post("/chat")
    async def chat(body: ChatRequest, _: Principal = Depends(require_scope("invoke"))):
        mgr = _require_module("model-manager")
        llm = _require_module("llm-serving")
        if not mgr.is_serving:
            raise HTTPException(status_code=503, detail="no model loaded")
        overrides = body.model_dump(exclude_none=True)
        messages = overrides.pop("messages")
        attachments = overrides.pop("attachments", None)
        if attachments:
            store = getattr(app, "ingest_store", None)
            if store is None:
                raise HTTPException(status_code=503, detail="ingestion unavailable")
            parts: list[str] = []
            for aid in attachments:
                try:
                    parts.append(store.load_text(aid, max_chars=settings.docqa_qa_max_context_chars))
                except Exception as exc: # noqa: BLE001 — UnknownIngestId / DocError → generic, no leak
                    if type(exc).__name__ == "UnknownIngestId":
                        raise HTTPException(status_code=400, detail="unknown attachment") from None
                    logger.warning("attachment load failed: %r", type(exc).__name__)
                    raise HTTPException(status_code=400, detail="attachment unreadable") from None
            context = "\n\n".join(parts)[: settings.docqa_qa_max_context_chars]
            messages = [{"role": "system",
                         "content": f"Reference the following attached document content:\n\n{context}"}] + messages
        if "max_tokens" in overrides:
            overrides["max_tokens"] = min(overrides["max_tokens"], settings.chat_max_tokens_ceiling)
        try:
            return await llm.chat(messages, **overrides)
        except Exception as exc: # noqa: BLE001 — upstream engine failure, no leak
            logger.warning("chat upstream error: %r", exc)
            raise HTTPException(status_code=502, detail="upstream engine error") from None

    # --- file ingestion: gated upload write-path. UNTRUSTED bytes → validate +
    # confine under `docs_root/_ingest/` (the IngestPolicy crux) → opaque id. `ingest`-scoped;
    # the only endpoint that adds backend write authority. Bounded read (no unbounded memory),
    # count/total caps, typed no-leak errors. The UI is zero-authority (POST + relay id only). ---
    @api.post("/ingest")
    async def ingest_file(file: UploadFile = File(...),
                          principal: Principal = Depends(require_scope("ingest"))):
        store = getattr(app, "ingest_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="ingestion unavailable")
        policy = store.policy
        # bounded read: at most max_file_bytes + 1 (the spooled body is read, never an unbounded
        # in-memory load); a longer file is rejected as 413 before any write.
        data = await file.read(policy.max_file_bytes + 1)
        if len(data) > policy.max_file_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        size = len(data)
        filename = file.filename or ""
        # admit→validate→resolve→write→register atomically (the lock serializes concurrent uploads
        # so the count/total caps cannot be raced).
        async with store.lock:
            try:
                prepared = policy.validate(filename, size)
                policy.check_admission(store.count(), store.total_bytes(), size)
                dest = policy.resolve_destination(prepared)
            except Exception as exc: # noqa: BLE001 — typed IngestError → fixed-detail status, no leak
                code = _INGEST_ERROR_STATUS.get(type(exc).__name__)
                if code is None:
                    raise
                raise HTTPException(status_code=code, detail=_INGEST_STATUS_DETAIL[code]) from None
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            store.register(prepared, size)
        logger.info("ingest by %r: ext=%s size=%d id=%s",
                    principal.subject, prepared.ext, size, prepared.ingest_id)
        return {"id": prepared.ingest_id, "filename": prepared.stored_name,
                "size": size, "ext": prepared.ext}

    # --- WebSocket: authenticated handshake (token via header or ?token=) ---
    @api.websocket("/ws")
    async def ws(websocket: WebSocket):
        principal = authenticate_ws(
            settings,
            api_key=websocket.headers.get("x-api-key"),
            authorization=websocket.headers.get("authorization"),
            token_qs=websocket.query_params.get("token"),
        )
        # Accept FIRST, then reject auth failures with a 1008 CLOSE FRAME. Closing *before* accept makes
        # the ASGI server answer the upgrade with HTTP 403, which a browser surfaces as an abnormal close
        # (code 1006) — indistinguishable from "server unreachable", so the client would reconnect-loop on
        # a bad token. Accepting then close(1008) delivers a real 1008 frame the client treats as
        # auth_failed (no reconnect). No data is ever sent to an unauthenticated socket.
        await websocket.accept()
        if principal is None:
            await websocket.close(code=1008) # policy violation — auth rejected
            return
        await websocket.send_json({"event": "capabilities",
                                   "available": sorted(app.available_capabilities())})
        # the agent bundle is accessed structurally — no import of the
        # orchestrator/session modules, preserving the gateway's no-core→modules-import rule.
        runtime = getattr(app, "agent_runtime", None)
        try:
            while True:
                raw = await websocket.receive_text()
                parsed = None
                try:
                    decoded = json.loads(raw)
                    if isinstance(decoded, dict):
                        parsed = decoded
                except (ValueError, TypeError):
                    parsed = None
                if parsed is not None and parsed.get("action") == "run_task":
                    if runtime is None:
                        await websocket.send_json({"event": "error", "reason": "agent not enabled"})
                        continue
                    # hand the connection to the agent session for the duration of the task
                    await runtime.run_ws_session(websocket, principal, parsed)
                    continue
                await websocket.send_json({"event": "echo", "data": raw})
        except WebSocketDisconnect:
            return

    return api
