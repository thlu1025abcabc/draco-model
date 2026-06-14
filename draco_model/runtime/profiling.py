from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Iterator

import polars as pl

from draco_model.core import Model, Node


_CACHE_CANDIDATE_OPS = {
    "_grid_project",
    "aggregate",
    "fill_null",
    "join",
    "op",
}


@dataclass(frozen=True)
class PlanNodeProfile:
    """Static profile for one structural node across a group of models."""

    node_id: str
    kind: str
    op: str
    params: dict[str, Any]
    ref_count: int
    model_count: int
    models: tuple[str, ...]
    depth: int
    is_shared: bool
    cache_candidate: bool


@dataclass(frozen=True)
class PlanProfile:
    """Static DAG profile used to find shared subgraphs before execution."""

    nodes: tuple[PlanNodeProfile, ...]

    def shared_nodes(self) -> tuple[PlanNodeProfile, ...]:
        """Return nodes referenced by more than one model DAG."""
        return tuple(node for node in self.nodes if node.is_shared)

    def cache_candidates(self) -> tuple[PlanNodeProfile, ...]:
        """Return shared nodes worth considering for batch materialization."""
        return tuple(node for node in self.nodes if node.cache_candidate)

    def summary(self) -> dict[str, int]:
        """Return stable counts for tests and lightweight diagnostics."""
        return {
            "node_count": len(self.nodes),
            "shared_node_count": len(self.shared_nodes()),
            "cache_candidate_count": len(self.cache_candidates()),
        }

    def to_frame(self) -> pl.DataFrame:
        """Return the profile as a Polars DataFrame."""
        return pl.DataFrame(
            [
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "op": node.op,
                    "params": _json_dumps(node.params),
                    "ref_count": node.ref_count,
                    "model_count": node.model_count,
                    "models": list(node.models),
                    "depth": node.depth,
                    "is_shared": node.is_shared,
                    "cache_candidate": node.cache_candidate,
                }
                for node in self.nodes
            ]
        )


@dataclass(frozen=True)
class ProfileEvent:
    """One runtime profiling event emitted while an Engine method runs."""

    event: str
    at_ms: float
    elapsed_ms: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)


class Profiler:
    """Collect lightweight runtime events without changing execution semantics."""

    def __init__(self) -> None:
        self._started_at = perf_counter()
        self._events: list[ProfileEvent] = []

    def record(self, event: str, *, elapsed_ms: float | None = None, **fields: Any) -> None:
        """Append one profiling event."""
        self._events.append(
            ProfileEvent(
                event=event,
                at_ms=(perf_counter() - self._started_at) * 1000,
                elapsed_ms=elapsed_ms,
                fields=dict(fields),
            )
        )

    @contextmanager
    def span(self, event: str, **fields: Any) -> Iterator[None]:
        """Record start/end events around a block."""
        self.record(f"{event}.start", **fields)
        start = perf_counter()
        try:
            yield
        except Exception as exc:
            self.record(
                f"{event}.error",
                elapsed_ms=(perf_counter() - start) * 1000,
                error_type=type(exc).__name__,
                **fields,
            )
            raise
        else:
            self.record(f"{event}.end", elapsed_ms=(perf_counter() - start) * 1000, **fields)

    def events(self) -> tuple[ProfileEvent, ...]:
        """Return collected events in emission order."""
        return tuple(self._events)

    def summary(self) -> dict[str, Any]:
        """Return event counts and the total profiled duration."""
        counts = Counter(event.event for event in self._events)
        return {
            "event_count": len(self._events),
            "elapsed_ms": (perf_counter() - self._started_at) * 1000,
            "counts": dict(counts),
        }

    def to_frame(self) -> pl.DataFrame:
        """Return runtime events as a Polars DataFrame."""
        return pl.DataFrame(
            [
                {
                    "event": event.event,
                    "at_ms": event.at_ms,
                    "elapsed_ms": event.elapsed_ms,
                    "model": _string_field(event, "model"),
                    "universe": _string_field(event, "universe"),
                    "date": _string_field(event, "date"),
                    "node_id": _string_field(event, "node_id"),
                    "op": _string_field(event, "op"),
                    "fields": _json_dumps(event.fields),
                }
                for event in self._events
            ]
        )


def profile_plan(models: Iterable[Model]) -> PlanProfile:
    """Build a static profile of shared structural nodes across models."""
    model_list = list(models)
    nodes_by_id: dict[str, Node] = {}
    models_by_id: dict[str, set[str]] = defaultdict(set)
    ref_counts: Counter[str] = Counter()
    depths: dict[str, int] = {}
    order: list[str] = []

    for model in model_list:
        local_depths: dict[str, int] = {}
        for node in model.nodes():
            if node.id not in nodes_by_id:
                nodes_by_id[node.id] = node
                order.append(node.id)
            ref_counts[node.id] += 1
            models_by_id[node.id].add(model.name)
            depth = _node_depth(node, local_depths)
            local_depths[node.id] = depth
            depths[node.id] = max(depths.get(node.id, depth), depth)

    return PlanProfile(
        tuple(
            _plan_node_profile(
                nodes_by_id[node_id],
                ref_counts[node_id],
                tuple(sorted(models_by_id[node_id])),
                depths[node_id],
            )
            for node_id in order
        )
    )


def _plan_node_profile(node: Node, ref_count: int, models: tuple[str, ...], depth: int) -> PlanNodeProfile:
    is_shared = ref_count > 1
    return PlanNodeProfile(
        node_id=node.id,
        kind=node.kind,
        op=node.op,
        params=dict(node.params),
        ref_count=ref_count,
        model_count=len(models),
        models=models,
        depth=depth,
        is_shared=is_shared,
        cache_candidate=is_shared and node.kind == "frame" and node.op in _CACHE_CANDIDATE_OPS,
    )


def _node_depth(node: Node, local_depths: dict[str, int]) -> int:
    if not node.inputs:
        return 0
    return max(local_depths[parent.id] for parent in node.inputs.values()) + 1


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _string_field(event: ProfileEvent, key: str) -> str | None:
    value = event.fields.get(key)
    return None if value is None else str(value)
