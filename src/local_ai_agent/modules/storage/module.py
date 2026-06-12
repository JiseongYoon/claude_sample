"""StorageModule — the registry-facing storage capability.

Loads connector configs (`storage_connectors_file`), builds a `GuardedConnector` per declared
connector (currently `kind="ssh"` → `SSHTransport`; Synology → sub-phase 5.1), and exposes the
gated `Tool`s (`.tools`) the composition root registers on the dispatcher. Connecting to
external hosts, it declares no in-registry dependency; `health` reflects connector *configuration*
(not a network probe — reachability is handled gracefully at tool-call time).
"""
from __future__ import annotations

from ...config import Settings
from ...core.module import Health, HealthStatus, ModuleSpec
from .config import StorageConnectorConfig, load_connector_configs
from .connector import GuardedConnector
from .ssh import SSHTransport
from .tools import build_tools


def _build_connector(cfg: StorageConnectorConfig) -> GuardedConnector:
    if cfg.kind == "ssh":
        transport = SSHTransport(cfg, max_bytes=cfg.max_bytes)
        return GuardedConnector(
            transport, allowed_root=cfg.allowed_root, read_only=cfg.read_only,
            max_bytes=cfg.max_bytes, realpath=transport.realpath, name=cfg.name,
        )
    raise ValueError(f"unsupported connector kind: {cfg.kind!r}") # synology → 5.1


class StorageModule:
    """`Module` exposing the `storage` capability (permissioned remote file access)."""

    def __init__(self, settings: Settings, chat: object | None = None, *,
                 connectors: dict[str, GuardedConnector] | None = None) -> None:
        # `connectors` is a test-only injection seam: pre-built `GuardedConnector`s over a
        # fake transport, so the real module/tool wiring runs without a live SSH server. None (production)
        # → load + build from the operator's `storage_connectors_file`.
        if connectors is not None:
            self._connectors: dict[str, GuardedConnector] = dict(connectors)
            # injected (test) path: no configs → project the non-secret fields the connector carries.
            self._views: list[dict] = [self._view_from_connector(c) for c in self._connectors.values()]
        else:
            if settings.storage_connectors_file is None:
                raise ValueError("storage enabled but storage_connectors_file is not configured")
            configs = load_connector_configs(settings.storage_connectors_file) # raises on bad config
            self._connectors = {cfg.name: _build_connector(cfg) for cfg in configs}
            self._views = [self._view_from_config(cfg) for cfg in configs]
        self._tools = build_tools(self._connectors)
        # when a ChatModel is available (enable_docqa), also expose the
        # remote → DocQA tools (`summarize_remote` / `answer_remote`) over the same connectors.
        if chat is not None:
            from ..docqa.tools import DocQAConfig
            from .remote_docqa import build_remote_tools

            self._tools += build_remote_tools(self._connectors, chat, DocQAConfig.from_settings(settings))
        self._started = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(
            name="storage", version="0.1.0",
            capabilities=("storage",), depends_on=(),
            description="permissioned remote file access (SSH/SFTP)",
        )

    @property
    def tools(self) -> list:
        return list(self._tools)

    @staticmethod
    def _view_from_config(cfg: StorageConnectorConfig) -> dict:
        """NON-SECRET projection of a connector config for the read-only listing endpoint.
        NEVER includes `auth` (key_path / password_env) or `known_hosts_path` (a local FS path)."""
        return {"name": cfg.name, "kind": cfg.kind, "host": cfg.host, "port": cfg.port,
                "username": cfg.username, "read_only": cfg.read_only, "allowed_root": cfg.allowed_root}

    @staticmethod
    def _view_from_connector(c: GuardedConnector) -> dict:
        """NON-SECRET projection from a pre-built connector (test-injection path) — only the
        fields the `GuardedConnector` carries; host/port/username are not retained there."""
        return {"name": c.name, "kind": "ssh", "host": None, "port": None,
                "username": None, "read_only": c.read_only, "allowed_root": c.allowed_root}

    def connector_views(self) -> list[dict]:
        """Read-only NON-SECRET view of the configured connectors. The module owns the
        no-secret rule (it knows which fields bear secrets); the gateway just serializes this."""
        return [dict(v) for v in self._views]

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        for conn in self._connectors.values():
            try:
                await conn.close()
            except Exception: # noqa: BLE001 — best-effort teardown
                pass
        self._started = False

    def health(self) -> Health:
        if not self._started:
            return Health(HealthStatus.absent, "not started")
        if not self._connectors:
            return Health(HealthStatus.absent, "no connectors configured")
        return Health(HealthStatus.ok, f"{len(self._connectors)} connector(s)")
