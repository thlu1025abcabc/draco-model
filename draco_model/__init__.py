"""Function/model-as-factor computation graph backed by Polars."""

from draco_model.core import Layer, Model, Node
from draco_model.recipes import FactorRecipe, Shortcut, metric, transform
from draco_model.runtime.engine import Engine
from draco_model.runtime.execution import TraceStep
from draco_model.runtime.profiling import PlanNodeProfile, PlanProfile, ProfileEvent, Profiler, profile_plan
from draco_model import layers as _layers  # noqa: F401  Ensure built-in executors/plans are registered.

__all__ = [
    "Engine",
    "FactorRecipe",
    "Layer",
    "Model",
    "Node",
    "PlanNodeProfile",
    "PlanProfile",
    "ProfileEvent",
    "Profiler",
    "Shortcut",
    "TraceStep",
    "metric",
    "profile_plan",
    "transform",
]
