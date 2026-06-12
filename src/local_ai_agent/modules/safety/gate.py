"""Safety gate — deterministic three-tier action classifier.

Pure code. No LLM, no network, no filesystem access (matching only — it never
executes a command or opens a path). The model *proposes* actions; this code
*disposes*. Fail-closed: nothing is `safe` unless it is explicitly allowlisted
and trips no blocked/needs-confirmation rule.

Three tiers (per requirements #7):
- blocked : catastrophic/forbidden — never run, never approvable.
- needs_confirmation : dangerous but allowable WITH human approval.
- safe : allowlisted read-only tools — auto-run.

Rules are supplied at construction (config/code), never mutated via an API.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Verdict(str, Enum):
    blocked = "blocked"
    needs_confirmation = "needs_confirmation"
    safe = "safe"


@dataclass(frozen=True)
class Action:
    """A proposed tool call to be classified."""

    tool: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""
    rule: str = ""

    @property
    def runnable_without_approval(self) -> bool:
        return self.verdict is Verdict.safe


@dataclass(frozen=True)
class Rule:
    name: str
    verdict: Verdict
    reason: str
    predicate: Callable[[Action], bool]


# --------------------------------------------------------------------------- #
# argument scanning (fail-closed: inspect EVERY string arg, not just known keys,
# so a path/command under an unexpected key name cannot bypass the rules)
# --------------------------------------------------------------------------- #
_CMD_KEYS = ("command", "cmd", "script", "shell", "code")


def _all_strings(value) -> list[str]:
    """Every string value anywhere in the args (recurses dict/list)."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out += _all_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            out += _all_strings(v)
    return out


def _joined_lists(value) -> list[str]:
    """For every list/tuple containing >1 string anywhere in the args, its space-joined form. An
    argv-style command (`["rm","-rf","/"]`) is thereby matchable by the string-form catastrophic /
    write-redirect patterns, so it is classified IDENTICALLY to the same command passed as one string.
    Without this, `_all_strings` yields the argv tokens separately and a multi-token catastrophic command
    slips past the block tier (it would only trip `confirm.shell`). Fail-closed hardening."""
    out: list[str] = []
    if isinstance(value, dict):
        for v in value.values():
            out += _joined_lists(v)
    elif isinstance(value, (list, tuple)):
        strs = [v for v in value if isinstance(v, str)]
        if len(strs) > 1:
            out.append(" ".join(strs))
        for v in value:
            out += _joined_lists(v)
    return out


def _canon(s: str) -> str:
    """expanduser + normpath, collapsing a leading run of slashes to one."""
    return re.sub(r"^/{2,}", "/", os.path.normpath(os.path.expanduser(s)))


def canonical_strings(args: dict) -> list[str]:
    """All string args, each canonicalized (so `../`, `~`, `//` are resolved)."""
    return [_canon(s) for s in _all_strings(args)]


def commands(args: dict) -> list[str]:
    """String values under command-ish keys (for the shell-presence heuristic)."""
    return [args[k] for k in _CMD_KEYS if isinstance(args.get(k), str)]


# --------------------------------------------------------------------------- #
# default rule predicates — all leading-slash-insensitive, case-insensitive
# --------------------------------------------------------------------------- #
# catastrophic shell patterns (blocked)
_BLOCK_CMD = re.compile(
    r"\brm\s+(?:-\S+\s+)+/(?![\w.~/-])" # rm <flags> / (root; any flags, any terminator incl ; | " *)
    r"|:\s*\(\s*\)\s*\{" # fork bomb :(){
    r"|\bmkfs\b|\bdd\s+if=" # disk wipe
    r"|>\s*/dev/sd" # write raw disk
    r"|\bchmod\s+-R\s+0*777\s+/" # chmod 777 /
    r"|\bshutdown\b|\breboot\b",
    re.IGNORECASE,
)
_WRITE_REDIRECT = re.compile(r">|>>|\btee\b|\bcp\b|\bmv\b|\bchmod\b|\bchown\b|\binstall\b", re.IGNORECASE)
# system path as a segment (matches /etc/x, //etc/x→/etc/x, ../../etc/x, etc/x)
_SYSTEM_RE = re.compile(r"(^|/)(etc|boot|sys|proc|dev|usr|opt|root|run)(/|$)", re.IGNORECASE)
# secret/credential paths — even reading is blocked
_SECRET_RE = re.compile(
    r"(^|/)etc/shadow(\b|$)"
    r"|(^|/)\.ssh(/|$)"
    r"|\.pem$|\.key$"
    r"|(^|/)\.env(rc)?(\.|/|$)"
    r"|(^|/)\.aws(/|$)"
    r"|id_rsa|id_dsa|id_ecdsa|id_ed25519"
    r"|(^|/)\.pgpass(\b|$)|(^|/)\.netrc(\b|$)"
    r"|\.git-credentials|(^|/)\.?credentials(\b|$)"
    r"|/proc/[^/]+/mem(\b|$)|(^|/)proc/kcore|/dev/k?mem(\b|$)"
    r"|docker\.sock(\b|$)|(^|/)run/secrets(/|$)",
    re.IGNORECASE,
)
# tampering with the agent/safety config (blocked)
_SAFETY_TAMPER = re.compile(r"(^|/)\.claude(/|$)|allowlist\.json|settings\.json", re.IGNORECASE)
# mutation command verbs (needs-confirmation)
_MUTATE_CMD = re.compile(
    r"\b(rm|rmdir|mv|cp|tee|truncate|install|pip\s+install|npm\s+install|"
    r"git\s+(push|reset|clean)|kill|chmod|chown|ln)\b|>\s*\S|>>\s*\S",
    re.IGNORECASE,
)
_WRITE_TOOL_RE = re.compile(
    r"write|save|create|append|delete|remove|overwrite|edit|move|rename|mkdir|touch|put|upload",
    re.IGNORECASE,
)


def _is_write_tool(tool: str) -> bool:
    return tool in _MUTATING_FILE_TOOLS or bool(_WRITE_TOOL_RE.search(tool))


def _any_secret_path(action: Action) -> bool:
    # scan ALL string args (any key) — reading a secret path is blocked outright
    return any(_SECRET_RE.search(s) for s in canonical_strings(action.args))


def _writes_system_path(action: Action) -> bool:
    # a write-intent tool touching a system path, OR a shell command redirecting
    # output into a system path
    if _is_write_tool(action.tool):
        if any(_SYSTEM_RE.search(s) for s in canonical_strings(action.args)):
            return True
    for cmd in _all_strings(action.args) + _joined_lists(action.args):
        if _SYSTEM_RE.search(_canon(cmd)) and _WRITE_REDIRECT.search(cmd):
            return True
    return False


def _safety_tamper(action: Action) -> bool:
    return any(_SAFETY_TAMPER.search(s) for s in canonical_strings(action.args))


def _nonlocal_path(action: Action) -> bool:
    """Any string arg that canonicalizes to an absolute path or escapes the
    workspace (`/...`, `~...`→`/...`, `../...`). The robust fail-safe behind the
    `safe` tier: even an allowlisted read tool only auto-runs on in-workspace
    relative paths — absolute/escaping targets always require confirmation. This
    closes the whole "absolute path to a sensitive location → safe" class without
    enumerating every sensitive prefix."""
    for s in canonical_strings(action.args):
        if s.startswith("/") or s.startswith(".."):
            return True
    return False


def _block_command(action: Action) -> bool:
    # scan each string arg AND the space-joined form of any multi-token list (argv) — so a catastrophic
    # command is blocked whether passed as "rm -rf /" or as ["rm","-rf","/"].
    return any(_BLOCK_CMD.search(s) for s in _all_strings(action.args) + _joined_lists(action.args))


_MUTATING_FILE_TOOLS = frozenset({
    "delete_file", "remove_file", "rm", "write_file", "overwrite_file",
    "move_file", "rename_file", "append_file", "edit_file", "create_file",
    "write_workspace_file", # exec: sandbox file write → needs_confirmation
})
_SHELL_TOOLS = frozenset({"shell", "exec", "run_command", "bash", "sh", "system"})
_EXTERNAL_POST_TOOLS = frozenset({"http_post", "http_put", "http_delete", "send_email",
                                  "webhook", "post"})
_NAV_TOOLS = frozenset({"navigate", "browser_navigate", "open_url", "download", "submit_form",
                        "web_answer"})
_STORAGE_MUTATE_TOOLS = frozenset({"storage_write", "storage_delete", "storage_rename",
                                   "storage_upload", "storage_move", "remote_write"})

DEFAULT_SAFE_TOOLS = frozenset({
    "read_file", "list_dir", "list_files", "stat", "search", "web_search",
    "http_get", "fetch_url", "calculator", "translate", "summarize",
    "get_weather", "get_stock_price",
})


def default_blocked_rules() -> list[Rule]:
    return [
        Rule("block.command", Verdict.blocked, "catastrophic shell command", _block_command),
        Rule("block.secret_read", Verdict.blocked, "access to a secret/credential path", _any_secret_path),
        Rule("block.system_write", Verdict.blocked, "write to a protected system path", _writes_system_path),
        Rule("block.safety_tamper", Verdict.blocked, "tampering with the agent/safety config", _safety_tamper),
    ]


def default_confirm_rules() -> list[Rule]:
    return [
        Rule("confirm.file_mutation", Verdict.needs_confirmation, "file create/modify/delete",
             lambda a: a.tool in _MUTATING_FILE_TOOLS),
        Rule("confirm.shell", Verdict.needs_confirmation, "side-effecting shell execution",
             lambda a: a.tool in _SHELL_TOOLS or bool(commands(a.args))),
        Rule("confirm.mutating_command", Verdict.needs_confirmation, "mutating command",
             lambda a: any(_MUTATE_CMD.search(c) for c in commands(a.args))),
        Rule("confirm.external_post", Verdict.needs_confirmation, "outbound side-effecting request",
             lambda a: a.tool in _EXTERNAL_POST_TOOLS),
        Rule("confirm.navigation", Verdict.needs_confirmation, "risky web navigation/download",
             lambda a: a.tool in _NAV_TOOLS),
        Rule("confirm.remote_mutation", Verdict.needs_confirmation, "remote-storage mutation",
             lambda a: a.tool in _STORAGE_MUTATE_TOOLS),
        # an external MCP server's tool (namespaced `mcp__<server>__<tool>`) is UNTRUSTED —
        # always confirm. The fail-safe default already does this; this rule makes it explicit and
        # robust if `DEFAULT_SAFE_TOOLS` ever changes (external tools are never safe-listed).
        Rule("confirm.mcp_external", Verdict.needs_confirmation, "external MCP server tool (untrusted)",
             lambda a: isinstance(a.tool, str) and a.tool.startswith("mcp__")),
        # touching any system path (read or otherwise) that wasn't already blocked
        # → require confirmation (e.g. relative `etc/x`). Fail-safe.
        Rule("confirm.system_path", Verdict.needs_confirmation, "references a system path",
             lambda a: any(_SYSTEM_RE.search(s) for s in canonical_strings(a.args))),
        # robust catch-all: an absolute or workspace-escaping path is never auto-safe.
        Rule("confirm.nonlocal_path", Verdict.needs_confirmation,
             "absolute or workspace-escaping path", _nonlocal_path),
    ]


class SafetyGate:
    """Deterministic classifier. Construct with custom rules/allowlist or defaults."""

    def __init__(self, *, blocked_rules: list[Rule] | None = None,
                 confirm_rules: list[Rule] | None = None,
                 safe_tools: frozenset[str] | None = None) -> None:
        self._blocked = blocked_rules if blocked_rules is not None else default_blocked_rules()
        self._confirm = confirm_rules if confirm_rules is not None else default_confirm_rules()
        self._safe_tools = safe_tools if safe_tools is not None else DEFAULT_SAFE_TOOLS

    def classify(self, action: Action) -> Decision:
        # 1. malformed → blocked (fail-closed; never raise)
        if not isinstance(action.tool, str) or not action.tool or not isinstance(action.args, dict):
            return Decision(Verdict.blocked, "malformed action", "malformed")
        try:
            # 2. blocked rules (highest precedence)
            for rule in self._blocked:
                if rule.predicate(action):
                    return Decision(Verdict.blocked, rule.reason, rule.name)
            # 3. needs-confirmation rules
            for rule in self._confirm:
                if rule.predicate(action):
                    return Decision(Verdict.needs_confirmation, rule.reason, rule.name)
            # 4. explicit safe allowlist
            if action.tool in self._safe_tools:
                return Decision(Verdict.safe, "allowlisted read-only tool", "safe.allowlist")
        except Exception as exc: # noqa: BLE001 — a buggy rule must fail closed
            return Decision(Verdict.needs_confirmation, f"classifier error: {exc!r}", "fail_closed")
        # 5. default fail-safe
        return Decision(Verdict.needs_confirmation, "unclassified action — default to confirmation", "default")
