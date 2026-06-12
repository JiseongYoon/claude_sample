"""Safety package: the deterministic HITL safety gate.

`gate.py` is the pure classifier — the single source of truth for whether
a proposed action is `safe` / `needs_confirmation` / `blocked`. It is code-only
(no LLM, no I/O), so a prompt-injected model cannot talk its way past it.
"""
