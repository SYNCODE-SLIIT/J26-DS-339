"""Harmony-oriented melody understanding: baseline implementation."""
# Expose the vocal analysis pipeline at the package level.
from .pipeline import analyze_vocal
# Declare the public API for wildcard imports.
__all__ = ["analyze_vocal"]
