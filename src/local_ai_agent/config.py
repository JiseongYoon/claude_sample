"""Configuration — the single typed source of config for the serving layer.

Loaded from environment variables / a `.env` file via pydantic-settings. Later phases
extend `Settings`; serving steps import `get_settings()` instead of hard-coding values.

This module is self-contained (no imports from other project modules) — per the
project's modularity / single-responsibility rule.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServeEngine(str, Enum):
    """Serving engine selection.

    `llamacpp` is the primary engine; `vllm` is kept as a selectable validation
    engine. `sglang` was dropped (no sm_89 gemma4 kernel) and is intentionally
    not a member — selecting it raises a ValidationError.
    """

    llamacpp = "llamacpp"
    vllm = "vllm"


class SplitMode(str, Enum):
    """llama.cpp multi-GPU split mode."""

    layer = "layer"
    row = "row"
    none = "none"


class Settings(BaseSettings):
    """Typed, validated configuration.

    Field names map to UPPER_SNAKE env vars (case-insensitive). Required fields with
    no default must be supplied via env / `.env`; everything else has a documented
    default. `protected_namespaces=()` lets us use the natural `model_*` field names
    without colliding with pydantic's reserved `model_` namespace.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    # --- model locations (required) ---
    model_safetensors_dir: Path
    model_gguf_dir: Path
    gguf_file: str | None = None
    # optional vision projector (multimodal). A bare `*.gguf` filename in MODEL_GGUF_DIR
    # (operator prerequisite — NOT in the repo; generated from the multimodal safetensors per the
    # deployment.md recipe). When set + present, llama-server launches with `--mmproj` and the
    # `multimodal` capability lights up; unset → text-only serving, the visual path cleanly absent.
    model_mmproj_file: str | None = None

    # --- serving (common) ---
    serve_engine: ServeEngine = ServeEngine.llamacpp
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000
    served_model_name: str = "gemma-4-31b-it"
    cuda_visible_devices: str = "0,1"

    # --- serving (llama.cpp / primary) ---
    n_gpu_layers: int = 999 # -ngl : offload all layers
    split_mode: SplitMode = SplitMode.layer
    tensor_split: str = "1,1" # --tensor-split : even across the two cards
    ctx_size: int = 16384 # -c : context window
    use_jinja: bool = True # --jinja : GGUF-embedded gemma4 chat template

    # --- serving (vLLM / validation engine only) ---
    tensor_parallel_size: int = 2
    dtype: str = "bfloat16"
    max_model_len: int | None = None

    # --- API gateway bind ---
    # `serve_port` is the MODEL server (llama-server / `openai_base_url`). The core API gateway (uvicorn)
    # binds `api_port` — they MUST differ, else the model-manager's `llama-server --port serve_port`
    # collides with the gateway. Default 8080 (gateway) vs 8000 (model).
    api_port: int = 8080

    # --- API gateway security ---
    # CORS: explicit allowlist only — never wildcard. Empty list = CORS disabled (deny).
    cors_allow_origins: list[str] = []
    # Host header allowlist (anti host-spoofing). Defaults to loopback.
    trusted_hosts: list[str] = ["127.0.0.1", "localhost", "testserver"]
    # Redact internal health detail (e.g. exception text, paths) from the public
    # API by default; flip on only for a trusted/authenticated operator view.
    expose_health_detail: bool = False

    # --- API auth ---
    # When True (default), all routes except the public set require a credential.
    # Secrets live in `.env` (never commit). If enabled with neither secret set,
    # protected routes deny (401) — secure by default.
    auth_enabled: bool = True
    api_key: str | None = None # static API key (full access) for trusted machine clients
    jwt_secret: str | None = None # HS256 signing secret for issued/scoped/expiring tokens
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # --- demo modules ---
    enable_demo_modules: bool = False
    demo_flaky_broken: bool = False # when demo modules on, start `flaky` broken

    # --- model stack ---
    enable_model_stack: bool = False
    chat_max_tokens_ceiling: int = 4096 # hard clamp on per-request max_tokens (DoS bound)

    # --- safety / HITL ---
    approval_timeout_seconds: float = 300.0 # pending approvals expire (fail-closed) after this

    # --- agent loop ---
    enable_agent: bool = False
    agent_max_steps: int = 12 # max model turns per run (INV-5)
    agent_max_tool_calls: int = 32 # cumulative tool calls per run (INV-5)
    agent_result_char_cap: int = 4096 # truncate each fed-back observation
    agent_deadline_seconds: float | None = None # optional wall-clock bound per run
    approval_decision_timeout_seconds: float = 120.0 # how long a WS approval prompt waits
    # multi-turn — `run_task.history` (client-supplied prior turns) is seeded into the
    # loop's messages, bounded so it cannot defeat INV-5's bounded-work intent / prompt-bloat DoS.
    agent_max_history_messages: int = 20 # keep at most the most-recent N seeded history messages
    agent_history_char_cap: int = 4096 # truncate each seeded history message's content
    # token streaming — max gap between SSE tokens before the stream errors (so a
    # stalled engine can't hang the run past the loop's bounds even when no wall-clock deadline is set).
    llm_stream_idle_timeout_s: float = 60.0

    # --- docqa ---
    enable_docqa: bool = False
    docs_root: Path | None = None # the sandbox root DocQA may read; None → reads nothing
    docqa_max_doc_bytes: int = 10_000_000 # per-document size cap (DoS bound)

    # --- docqa chunker ---
    # chunk_chars = (ctx_size − prompt_reserve_tokens) × chars_per_token (char heuristic, no tiktoken).
    docqa_prompt_reserve_tokens: int = 2048 # tokens held back for prompt + model output
    docqa_chars_per_token: float = 3.0 # conservative (Korean/mixed dense; 4.0 = English-only)
    docqa_chunk_overlap_ratio: float = 0.1 # overlap chars = floor(chunk_chars × ratio); [0, 1)
    docqa_max_chunks: int = 64 # hard cap on chunks/doc (bounds downstream LLM use)

    # --- docqa summarizer ---
    docqa_summary_max_tokens: int = 512 # per-call output cap (bounds map/reduce inputs)
    docqa_reduce_max_passes: int = 5 # hard cap on hierarchical reduce passes (no runaway)

    # --- docqa QA ---
    docqa_qa_top_k: int = 5 # max chunks retrieved into the answer prompt
    docqa_qa_max_context_chars: int = 12000 # total char budget for retrieved context
    docqa_answer_max_tokens: int = 512 # per-answer output cap
    # QA retrieval chunk size — deliberately smaller than the (derived) summarize chunk size so
    # several chunks fit the QA context budget and retrieval has finer granularity.
    docqa_qa_chunk_chars: int = 2000

    # --- docqa tools ---
    docqa_max_docs_per_query: int = 20 # cap docs loaded for a directory query
    docqa_max_total_chunks: int = 128 # cap total chunks gathered for a directory query

    # --- visual multimodal ---
    # Bounds for the UNTRUSTED-PDF rasterizer (the one new server-side parser surface): a huge / bomb
    # PDF is refused, never an OOM/hang. Image bytes are NOT decoded server-side (base64 passthrough).
    multimodal_max_pdf_pages: int = 20 # cap pages rasterized from a scan-PDF
    multimodal_max_page_px: int = 4_000_000 # per-page pixel-count cap (zoom down to stay under)
    multimodal_max_render_bytes: int = 50_000_000 # total PNG bytes across rendered pages
    multimodal_render_timeout_s: float = 30.0 # wall-clock backstop across pages

    # --- file ingestion ---
    # An authenticated upload writes an UNTRUSTED file into a dedicated `_ingest/` area under
    # `docs_root` (DF1), referenced by an opaque id. Caps bound DoS; the endpoint is `ingest`-scoped.
    ingest_max_file_bytes: int = 10_000_000 # per-file size cap (defaults to docqa_max_doc_bytes)
    ingest_max_files: int = 50 # max stored ingested files (count cap)
    ingest_max_total_bytes: int = 200_000_000 # max total bytes across stored ingested files

    # --- storage ---
    enable_storage: bool = False
    storage_connectors_file: Path | None = None # JSON declaring connectors; None → none configured
    storage_max_bytes: int = 10_000_000 # default per-read size cap (connector may override)

    # --- browser / web ---
    enable_browser: bool = False
    browser_searxng_url: str | None = None # SearXNG endpoint (operator-trusted); None → search disabled
    browser_allow_hosts: list[str] = Field(default_factory=list) # empty → blocklist mode; non-empty → strict allowlist
    browser_max_bytes: int = 5_000_000 # per-fetch response size cap
    browser_per_fetch_timeout: float = 10.0 # per-request time budget (seconds)
    browser_total_timeout: float = 30.0 # total wall-clock budget for a multi-fetch synthesize
    browser_max_redirects: int = 5 # bounded redirect chain (each hop re-contained)
    browser_top_n: int = 5 # max search results fetched into a synthesize
    browser_max_text_chars: int = 20000 # cap on extracted text returned by a fetch/open tool

    # --- exec / sandboxed execution ---
    enable_exec: bool = False
    exec_image: str = "local_ai_agent_exec:base" # the hardened sandbox image (operator-built)
    exec_workspace_root: str = "/workspace" # in-container workspace mount point (NOT a host path)
    exec_network: str = "none" # v1 = no egress (installs/egress → sub-phase 7.1)
    exec_user: str = "1000:1000" # non-root uid:gid inside the sandbox
    exec_mem_bytes: int = 2_000_000_000 # container memory cap (== memory-swap, no swap escape)
    exec_cpus: float = 2.0 # CPU quota
    exec_pids_limit: int = 256 # fork-bomb cap (cgroup pids)
    exec_timeout: float = 30.0 # default per-command wall-clock budget (seconds)
    exec_max_output_bytes: int = 1_000_000 # stdout/stderr (and file-read) byte cap
    exec_ulimit_nofile: int = 1024 # open-fd cap
    exec_ulimit_fsize: int = 50_000_000 # max single-file size written (bytes)
    exec_tmpfs_size_bytes: int = 268_435_456 # /tmp tmpfs size (noexec,nosuid)
    exec_env_whitelist: list[str] = Field(default_factory=list) # only these env vars pass into the sandbox

    # --- mcp / remote tools ---
    # An MCP *client* consuming external servers' tools as GATED agent tools. The server is an
    # UNTRUSTED external party: every external tool routes through the gate (default confirm, never
    # safe-listed); tool descriptions/results are bounded + treated as untrusted input. v1 = stdio
    # transport (HTTP/SSE → sub-phase 8.1). Servers are operator-declared (the model never spawns one).
    enable_mcp: bool = False
    mcp_servers_file: Path | None = None # operator JSON declaring MCP servers; None → none configured
    mcp_call_timeout: float = 30.0 # per tool-call wall-clock budget (seconds)
    mcp_connect_timeout: float = 20.0 # per-server connect/handshake budget (seconds)
    mcp_max_result_bytes: int = 1_000_000 # tool-result content byte cap (untrusted-output bound)
    mcp_max_description_chars: int = 4000 # tool-description char cap (tool-poisoning bound)

    # --- presentation/avatar hostability demo ---
    # NOT a pre-built avatar seam (§A5). A minimal example proving the GENERIC Module interface + the
    # existing REST/WS API host a future presentation/Live2D module with ZERO core change.
    enable_presentation_demo: bool = False

    @field_validator("serve_port", "api_port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be in 1..65535")
        return v

    @model_validator(mode="after")
    def _gateway_and_model_ports_distinct(self) -> "Settings":
        # the gateway (api_port) and the model server (serve_port) run on the same host → distinct ports.
        if self.api_port == self.serve_port:
            raise ValueError("api_port (gateway) must differ from serve_port (model server)")
        return self

    @field_validator("tensor_parallel_size")
    @classmethod
    def _tp_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        return v

    @field_validator("ctx_size")
    @classmethod
    def _ctx_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("ctx_size must be >= 1")
        return v

    @field_validator("approval_timeout_seconds", "approval_decision_timeout_seconds")
    @classmethod
    def _timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout must be > 0")
        return v

    @field_validator("agent_max_steps", "agent_max_tool_calls", "agent_result_char_cap")
    @classmethod
    def _agent_bounds_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("agent bounds must be >= 0")
        return v

    @field_validator("agent_max_history_messages", "agent_history_char_cap")
    @classmethod
    def _agent_history_caps_positive(cls, v: int) -> int:
        # strictly positive (NOT the >= 0 of _agent_bounds_nonneg): a zero history cap would be a
        # foot-gun (silently no multi-turn / zero-length messages).
        if v <= 0:
            raise ValueError("agent history caps must be > 0")
        return v

    @field_validator("llm_stream_idle_timeout_s")
    @classmethod
    def _stream_idle_timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("llm_stream_idle_timeout_s must be > 0")
        return v

    @field_validator("docqa_max_doc_bytes")
    @classmethod
    def _docqa_bytes_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("docqa_max_doc_bytes must be > 0")
        return v

    @field_validator("multimodal_max_pdf_pages", "multimodal_max_page_px",
                     "multimodal_max_render_bytes", "multimodal_render_timeout_s")
    @classmethod
    def _multimodal_caps_positive(cls, v):
        if v <= 0:
            raise ValueError("multimodal caps must be > 0")
        return v

    @field_validator("ingest_max_file_bytes", "ingest_max_files", "ingest_max_total_bytes")
    @classmethod
    def _ingest_caps_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ingest caps must be > 0")
        return v

    @field_validator("docqa_prompt_reserve_tokens")
    @classmethod
    def _docqa_reserve_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("docqa_prompt_reserve_tokens must be >= 0")
        return v

    @field_validator("docqa_chars_per_token")
    @classmethod
    def _docqa_cpt_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("docqa_chars_per_token must be > 0")
        return v

    @field_validator("docqa_chunk_overlap_ratio")
    @classmethod
    def _docqa_overlap_ratio_range(cls, v: float) -> float:
        if not 0 <= v < 1:
            raise ValueError("docqa_chunk_overlap_ratio must be in [0, 1)")
        return v

    @field_validator("docqa_max_chunks")
    @classmethod
    def _docqa_max_chunks_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("docqa_max_chunks must be > 0")
        return v

    @field_validator("docqa_summary_max_tokens", "docqa_reduce_max_passes")
    @classmethod
    def _docqa_summarizer_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("docqa summarizer bounds must be > 0")
        return v

    @field_validator(
        "docqa_qa_top_k", "docqa_qa_max_context_chars", "docqa_answer_max_tokens",
        "docqa_qa_chunk_chars", "docqa_max_docs_per_query", "docqa_max_total_chunks",
    )
    @classmethod
    def _docqa_qa_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("docqa QA/tool bounds must be > 0")
        return v

    @field_validator("storage_max_bytes")
    @classmethod
    def _storage_bytes_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("storage_max_bytes must be > 0")
        return v

    @field_validator(
        "browser_max_bytes", "browser_per_fetch_timeout", "browser_total_timeout",
        "browser_top_n", "browser_max_text_chars",
    )
    @classmethod
    def _browser_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("browser bounds must be > 0")
        return v

    @field_validator("browser_max_redirects")
    @classmethod
    def _browser_redirects_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("browser_max_redirects must be >= 0")
        return v

    @field_validator(
        "exec_mem_bytes", "exec_cpus", "exec_pids_limit", "exec_timeout",
        "exec_max_output_bytes", "exec_ulimit_nofile", "exec_ulimit_fsize", "exec_tmpfs_size_bytes",
    )
    @classmethod
    def _exec_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("exec bounds must be > 0")
        return v

    @field_validator("exec_image", "exec_user")
    @classmethod
    def _exec_str_non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("exec_image / exec_user must be non-empty")
        return v

    @field_validator("exec_workspace_root")
    @classmethod
    def _exec_workspace_absolute(cls, v: str) -> str:
        if not str(v).startswith("/"):
            raise ValueError("exec_workspace_root must be an absolute (in-container) path")
        return v

    @field_validator("exec_network")
    @classmethod
    def _exec_network_none_v1(cls, v: str) -> str:
        # v1 is no-egress; widening this is sub-phase 7.1 (out-of-process egress proxy + netns).
        if v != "none":
            raise ValueError("exec_network must be 'none' in v1 (egress → sub-phase 7.1)")
        return v

    @field_validator(
        "mcp_call_timeout", "mcp_connect_timeout", "mcp_max_result_bytes", "mcp_max_description_chars",
    )
    @classmethod
    def _mcp_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("mcp bounds must be > 0")
        return v

    @field_validator("model_safetensors_dir", "model_gguf_dir")
    @classmethod
    def _path_non_empty(cls, v: Path) -> Path:
        if str(v).strip() == "":
            raise ValueError("model path must not be empty")
        return v

    @property
    def openai_base_url(self) -> str:
        """OpenAI-compatible base URL clients should target."""
        return f"http://{self.serve_host}:{self.serve_port}/v1"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads env / `.env` once)."""
    return Settings()
