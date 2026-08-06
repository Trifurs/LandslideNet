"""Shared implementation for the three LandslideNet project entry scripts."""

from models import LandslideNet

from .model_registry import (
    MODEL_GROUPS,
    MODEL_SPECS,
    build_deep_model,
    canonical_model_name,
    expand_model_selection,
)

__all__ = [
    "LandslideNet",
    "MODEL_GROUPS",
    "MODEL_SPECS",
    "build_deep_model",
    "canonical_model_name",
    "expand_model_selection",
]
__version__ = "2.2.0"
