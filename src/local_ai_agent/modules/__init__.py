"""Capability-module seam.

Each capability (DocQA, Browser, Task/Exec, Storage connectors, Model Manager,
the future presentation/Live2D, …) lives here as its own single-responsibility
module implementing `local_ai_agent.core.module.Module`. Modules do not import
one another; the composition root (`main`) wires them and the registry
resolves their soft dependencies by health. Empty until phase 4+.
"""
