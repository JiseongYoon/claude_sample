"""Core platform internals — the composition machinery, not a capability.

Holds the `Module` seam, the application container, and (later phases) the module
registry and API gateway. Capability modules live under `local_ai_agent.modules`
and implement the `Module` protocol defined here; they never import each other.
"""
