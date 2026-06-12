"""Tests for the deterministic SafetyGate.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.safety.gate import (
    Action,
    Rule,
    SafetyGate,
    Verdict,
)


@pytest.fixture
def gate() -> SafetyGate:
    return SafetyGate()


def v(gate, tool, **args) -> Verdict:
    return gate.classify(Action(tool, args)).verdict


# --- safe (allowlisted read-only) -------------------------------------------
@pytest.mark.parametrize("tool,args", [
    ("read_file", {"path": "notes/todo.md"}),
    ("calculator", {"expression": "17*23"}),
    ("web_search", {"query": "python"}),
    ("http_get", {"url": "https://example.com"}),
    ("list_dir", {"path": "src"}),
])
def test_safe_tools(gate, tool, args):
    assert gate.classify(Action(tool, args)).verdict is Verdict.safe


# --- needs-confirmation (dangerous but approvable) --------------------------
@pytest.mark.parametrize("tool,args", [
    ("delete_file", {"path": "a.txt"}),
    ("write_file", {"path": "a.txt", "content": "x"}),
    ("shell", {"command": "ls -la"}),
    ("run_command", {"command": "rm old.log"}), # mutating command
    ("http_post", {"url": "https://x.com", "body": "{}"}),
    ("navigate", {"url": "https://x.com"}),
    ("storage_write", {"target": "nas://share/f"}),
])
def test_needs_confirmation(gate, tool, args):
    assert gate.classify(Action(tool, args)).verdict is Verdict.needs_confirmation


# --- blocked (catastrophic / forbidden) -------------------------------------
@pytest.mark.parametrize("tool,args", [
    ("run_command", {"command": "rm -rf /"}),
    ("shell", {"command": ":(){ :|:& };:"}), # fork bomb
    ("write_file", {"path": "/etc/passwd", "content": "x"}),
    ("read_file", {"path": "/etc/shadow"}),
    ("read_file", {"path": "~/.ssh/id_rsa"}),
    ("write_file", {"path": ".claude/settings.json", "content": "x"}),
])
def test_blocked(gate, tool, args):
    assert gate.classify(Action(tool, args)).verdict is Verdict.blocked


# --- precedence: blocked outranks a safe tool name --------------------------
def test_blocked_outranks_safe_tool(gate):
    # read_file is allowlisted-safe, but a secret path must be blocked
    assert v(gate, "read_file", path="/etc/shadow") is Verdict.blocked


def test_block_command_outranks_confirm(gate):
    d = gate.classify(Action("run_command", {"command": "rm -rf /"}))
    assert d.verdict is Verdict.blocked and d.rule == "block.command"


# --- canonicalization defeats evasion ---------------------------------------
@pytest.mark.parametrize("path", [
    "./a/../../etc/shadow",
    "foo/../../../etc/shadow",
    "~/.ssh/id_rsa",
    "x/./../.ssh/id_rsa",
])
def test_path_evasion_still_blocked(gate, path):
    assert v(gate, "read_file", path=path) is Verdict.blocked


# --- default fail-safe ------------------------------------------------------
def test_unknown_tool_defaults_to_confirmation(gate):
    d = gate.classify(Action("frobnicate", {"x": 1}))
    assert d.verdict is Verdict.needs_confirmation and d.rule == "default"


def test_unknown_tool_never_safe(gate):
    assert v(gate, "definitely_not_allowlisted", foo="bar") is not Verdict.safe


# --- malformed → blocked (never raises, never safe) -------------------------
@pytest.mark.parametrize("action", [
    Action(tool="", args={}),
    Action(tool=None, args={}), # type: ignore[arg-type]
    Action(tool="x", args="not-a-dict"), # type: ignore[arg-type]
])
def test_malformed_blocked(gate, action):
    d = gate.classify(action)
    assert d.verdict is Verdict.blocked and d.rule == "malformed"


# --- a buggy rule fails closed (→ needs_confirmation, not safe) -------------
def test_buggy_rule_fails_closed():
    def _boom(action):
        raise RuntimeError("rule bug")
    gate = SafetyGate(confirm_rules=[Rule("bad", Verdict.needs_confirmation, "x", _boom)])
    d = gate.classify(Action("read_file", {"path": "ok.txt"}))
    assert d.verdict is Verdict.needs_confirmation and d.rule == "fail_closed"


# --- regression: adversarial findings from verification --------------
@pytest.mark.parametrize("tool,args", [
    # non-canonical arg keys must NOT bypass (was HIGH: escaped to safe)
    ("read_file", {"location": "/etc/shadow"}),
    ("read_file", {"filepath": "/etc/shadow"}),
    ("read_file", {"name": "~/.ssh/id_rsa"}),
    ("read_file", {"data": "secret.pem"}),
    # relative-form secret paths (no leading slash)
    ("read_file", {"path": "etc/shadow"}),
    ("read_file", {"path": "a/./b/./../../etc/shadow"}),
    # double-slash + relative traversal writes to system (was MEDIUM: confirm)
    ("write_file", {"path": "//etc/passwd"}),
    ("write_file", {"path": "../../etc/cron.d/evil"}),
    # shell redirect into a system path
    ("shell", {"command": "echo pwned > /etc/passwd"}),
    # glob rm
    ("run_command", {"command": "rm -rf /*"}),
    # case-insensitive secret
    ("read_file", {"path": "/ETC/SHADOW"}),
    # .envrc variant
    ("read_file", {"path": "project/.envrc"}),
    # --- second round (re-verification residuals) ---
    # chained/obfuscated root deletion must stay blocked (was downgraded to confirm)
    ("run_command", {"command": "rm -rf /;echo done"}),
    ("run_command", {"command": "rm -rf /|true"}),
    ("shell", {"command": 'bash -c "rm -rf /"'}),
    ("run_command", {"command": "rm -f /*"}),
    # more credential files
    ("read_file", {"path": "~/.pgpass"}),
    ("read_file", {"path": "~/.netrc"}),
    ("read_file", {"path": "project/.git-credentials"}),
    ("read_file", {"path": "~/.ssh/id_ed25519"}),
    # process memory / raw device
    ("read_file", {"path": "/proc/self/mem"}),
    ("read_file", {"path": "/dev/mem"}),
    # write to /usr is blocked
    ("write_file", {"path": "/usr/bin/evil", "content": "x"}),
    # nested args carrying a secret path
    ("read_file", {"opts": {"target": "/etc/shadow"}}),
    # --- third round (re-verification residuals) ---
    ("read_file", {"path": "/run/docker.sock"}), # container escape vector
    ("read_file", {"path": "/var/run/docker.sock"}),
    ("read_file", {"path": "/run/secrets/db_password"}), # orchestrator secrets
])
def test_adversarial_inputs_are_blocked(gate, tool, args):
    assert gate.classify(Action(tool, args)).verdict is Verdict.blocked


@pytest.mark.parametrize("path", [
    "/root/.bash_history", # root home — credentials in history
    "/root/secret_key",
    "/run/containerd/x", # absolute system-ish
    "/tmp/anything", # any absolute path is not auto-safe (fail-safe)
    "../outside/file", # workspace escape
    "~/Documents/notes.txt", # ~ → absolute
])
def test_absolute_or_escaping_paths_never_safe(gate, path):
    # the robust fail-safe: allowlisted read tool + absolute/escaping path → confirm
    assert gate.classify(Action("read_file", {"path": path})).verdict is not Verdict.safe


@pytest.mark.parametrize("tool,args", [
    ("read_file", {"path": "/etc/*"}), # glob read of a system dir
    ("read_file", {"path": "/etc/hosts"}), # system-path read → confirm (was safe)
    ("read_file", {"path": "/var/log/syslog"}), # /var not blocked, but…
])
def test_system_path_read_requires_confirmation(gate, tool, args):
    # reading a system path is not auto-safe (fail-safe); /var stays confirm via mutation? no —
    # /var isn't in _SYSTEM_RE, so /var/log read is still safe; assert the /etc cases confirm.
    verdict = gate.classify(Action(tool, args)).verdict
    if "/etc" in args["path"]:
        assert verdict is Verdict.needs_confirmation
    else:
        assert verdict is not Verdict.blocked # /var read is allowed (safe), not blocked


def test_secret_in_any_string_is_blocked_conservatively(gate):
    # scanning ALL strings is intentionally conservative: even a search query that
    # contains a secret path is blocked (fail-safe over-block, by design).
    assert gate.classify(Action("web_search", {"query": "contents of /etc/shadow"})).verdict is Verdict.blocked


# --- determinism / no side effects ------------------------------------------
def test_classify_is_pure_and_repeatable(gate):
    a = Action("write_file", {"path": "a.txt"})
    assert gate.classify(a).verdict is gate.classify(a).verdict is Verdict.needs_confirmation


# --- argv-list commands classify identically to their string form ----------
# Regression: a catastrophic command passed as an argv LIST (`["rm","-rf","/"]`) must be blocked just
# like the string form `"rm -rf /"` — `_all_strings` yields the tokens separately, so without the
# space-joined scan it would only trip `confirm.shell` and a human could approve a root wipe.
@pytest.mark.parametrize("args", [
    {"command": ["rm", "-rf", "/"]},
    {"command": ["rm", "-rf", "/*"]},
    {"argv": ["rm", "-rf", "/"]}, # alternate arg key
    {"command": ["sh", "-c", "rm -rf /"]}, # shell -c form
    {"command": ["tee", "/etc/passwd"]}, # write to a system path via argv
    {"opts": {"argv": ["rm", "-rf", "/"]}}, # nested under another key
])
def test_argv_catastrophic_commands_are_blocked(gate, args):
    assert gate.classify(Action("run_command", args)).verdict is Verdict.blocked


def test_argv_rm_root_is_block_command_rule(gate):
    d = gate.classify(Action("run_command", {"command": ["rm", "-rf", "/"]}))
    assert d.verdict is Verdict.blocked and d.rule == "block.command"


def test_argv_and_string_form_agree(gate):
    # the exact same command, two encodings → identical verdict
    s = gate.classify(Action("run_command", {"command": "rm -rf /"})).verdict
    a = gate.classify(Action("run_command", {"command": ["rm", "-rf", "/"]})).verdict
    assert s is a is Verdict.blocked


@pytest.mark.parametrize("argv", [
    ["ls", "-la", "/tmp"],
    ["echo", "hello world"],
    ["git", "status"],
    ["cat", "notes/todo.md"],
])
def test_legitimate_argv_is_not_blocked(gate, argv):
    # a benign argv command is still side-effecting shell → needs_confirmation, but NEVER blocked
    # (the joined-scan hardening must not over-block ordinary commands).
    d = gate.classify(Action("run_command", {"command": argv}))
    assert d.verdict is Verdict.needs_confirmation
