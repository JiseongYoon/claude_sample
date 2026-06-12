"""Exec capability config.

A small frozen view over the main `Settings`, consumed by `ExecPolicy`/the exec module (mirrors
docqa's `DocQAConfig` and browser's `BrowserConfig`). `workspace_root` is the *in-container* mount
point of the per-task workspace volume — never a host path (host binds are forbidden by design, F1).
`network` is validated to `"none"` in v1 (no egress; package installs / general egress = sub-phase
7.1). Defaults are set so direct construction in tests stays terse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING: # avoid an import cycle / coupling at runtime
    from ...config import Settings


@dataclass(frozen=True)
class ExecConfig:
    """Resolved exec settings. All numeric bounds are > 0 (validated on `Settings`)."""

    image: str
    workspace_root: str # in-container path (e.g. /workspace); NOT a host path
    network: str # "none" in v1 (no egress)
    user: str # non-root, e.g. "1000:1000"
    mem_bytes: int
    cpus: float
    pids_limit: int
    timeout: float # default per-command wall-clock budget (seconds)
    max_output_bytes: int # stdout/stderr (and file-read) byte cap
    ulimit_nofile: int
    ulimit_fsize: int
    tmpfs_size_bytes: int
    env_whitelist: tuple[str, ...]

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ExecConfig":
        return cls(
            image=settings.exec_image,
            workspace_root=settings.exec_workspace_root,
            network=settings.exec_network,
            user=settings.exec_user,
            mem_bytes=settings.exec_mem_bytes,
            cpus=settings.exec_cpus,
            pids_limit=settings.exec_pids_limit,
            timeout=settings.exec_timeout,
            max_output_bytes=settings.exec_max_output_bytes,
            ulimit_nofile=settings.exec_ulimit_nofile,
            ulimit_fsize=settings.exec_ulimit_fsize,
            tmpfs_size_bytes=settings.exec_tmpfs_size_bytes,
            env_whitelist=tuple(settings.exec_env_whitelist or ()),
        )
