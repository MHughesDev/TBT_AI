"""TBT tank pricing — v5.

A physics-anchored, comparables-augmented ensemble with a decision layer matched
to mean absolute percentage error. See ../DESIGN.md for what is different from v4
and why, and ../README.md for how to run it.
"""
from .config import VERSION
from .data import ScopeError, load
from .model import TBT5
from .evaluate import rolling_origin, metrics

__all__ = ["VERSION", "ScopeError", "load", "TBT5", "rolling_origin", "metrics"]
