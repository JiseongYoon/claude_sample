"""Presentation/avatar hostability demo.

NOT a pre-built avatar seam. A minimal example proving the generic `Module` interface + the existing
REST/WS API can host a future presentation/Live2D module with ZERO core change.
"""
from .module import PresentationModule

__all__ = ["PresentationModule"]
