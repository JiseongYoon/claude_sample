"""Model-manager module package.

`process.py` owns the engine subprocess lifecycle; later steps wrap it as
a `Module` and expose it over the gateway. The serving *client* is a separate
module (`llm_serving`) that soft-depends on this one.
"""
