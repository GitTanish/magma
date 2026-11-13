"""Typed multi-graph store for agent memory.

One set of nodes, four relations over it:

* ``semantic`` -- bidirectional similarity links between event nodes
* ``temporal`` -- directed links from an earlier event to a later one
* ``causal``   -- directed links from cause to effect
* ``entity``   -- links between events and the entities they mention

This module owns topology and invariants only. Similarity scoring, edge
inference, and retrieval policy live in the layers above it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np


class NodeKind(str, Enum):
    """The two kinds of nodes in the memory graph."""

    EVENT = "event"
    ENTITY = "entity"


class EdgeType(str, Enum):
    """The four relational views described in the paper."""

    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    ENTITY = "entity"


class DuplicateNodeError(ValueError):
    """Raised when a node id is reused."""


class NodeNotFoundError(KeyError):
    """Raised when a node id is not present in the graph."""


class TemporalOrderError(ValueError):
    """Raised when a temporal edge would point backwards in time."""


@dataclass(slots=True)
class MemoryNode:
    """A single memory item.

    Event nodes represent observations or utterances and require a numeric
    timestamp. Entity nodes represent people, places, or objects and are shared
    by every event that mentions them.
    """

    id: str
    kind: NodeKind
    content: str
    timestamp: float | None = None
    embedding: np.ndarray | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("node id must be a non-empty string")
        self.id = self.id.strip()

        self.kind = NodeKind(self.kind)
        if not isinstance(self.content, str):
            raise TypeError("node content must be a string")

        if self.kind is NodeKind.EVENT and self.timestamp is None:
            raise ValueError(f"event node {self.id!r} requires a timestamp")

        if self.timestamp is not None:
            timestamp = float(self.timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("timestamp must be finite")
            self.timestamp = timestamp

        if self.embedding is not None:
            embedding = np.array(self.embedding, dtype=np.float32, copy=True)
            if embedding.ndim != 1 or embedding.size == 0:
                raise ValueError("embedding must be a non-empty 1-D vector")
            if not np.isfinite(embedding).all():
                raise ValueError("embedding must contain only finite values")
            self.embedding = embedding

        self.attributes = dict(self.attributes or {})

    @classmethod
    def event(
        cls,
        id: str,
        content: str,
        timestamp: float,
        *,
        embedding: np.ndarray | Iterable[float] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> MemoryNode:
        return cls(
            id=id,
            kind=NodeKind.EVENT,
            content=content,
            timestamp=timestamp,
            embedding=embedding,
            attributes=dict(attributes or {}),
        )

    @classmethod
    def entity(
        cls,
        id: str,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> MemoryNode:
        return cls(
            id=id,
            kind=NodeKind.ENTITY,
            content=name,
            attributes=dict(attributes or {}),
        )

    def __repr__(self) -> str:
        label = self.content
        if len(label) > 48:
            label = f"{label[:45]}..."
        return (
            f"MemoryNode(id={self.id!r}, kind={self.kind.value!r}, "
            f"content={label!r}, timestamp={self.timestamp!r})"
        )


@dataclass(frozen=True, slots=True)
class MemoryEdge:
    """A directed, typed, weighted edge between two memory nodes."""

    source: str
    target: str
    type: EdgeType
    weight: float
    attributes: Mapping[str, Any] = field(default_factory=dict)


class MemoryGraph:
    """A directed multigraph over event and entity nodes.

    Multiple relations may connect the same pair of nodes, so the underlying
    container is a ``MultiDiGraph`` keyed by edge type. Weights are normalized
    relevance or confidence values in ``(0, 1]``; the retrieval layer can then
    combine them without worrying about incompatible scales.

    Temporal edges must point from an earlier event to a later event. Semantic
    and entity links are stored in both directions so traversal can move either
    way without special cases.
    """

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: dict[str, MemoryNode] = {}
        self._entity_index: dict[str, str] = {}
        self._latest_event_id: str | None = None
        self._event_sequence = 0
        self._entity_sequence = 0

    # ------------------------------------------------------------------
    # Node access
    # ------------------------------------------------------------------

    def add_node(self, node: MemoryNode) -> MemoryNode:
        """Insert a node, rejecting duplicate ids and duplicate entity names."""
        if not isinstance(node, MemoryNode):
            raise TypeError("add_node expects a MemoryNode")

        if node.id in self._nodes:
            raise DuplicateNodeError(f"node id {node.id!r} already exists")

        entity_key: str | None = None
        if node.kind is NodeKind.ENTITY:
            entity_key = _normalize_name(node.content)
            existing = self._entity_index.get(entity_key)
            if existing is not None:
                raise DuplicateNodeError(
                    f"entity {node.content!r} already exists as {existing!r}; "
                    "reuse it or resolve the alias explicitly"
                )

        self._nodes[node.id] = node
        self._graph.add_node(node.id)

        if node.kind is NodeKind.EVENT:
            if self._latest_event_id is None:
                self._latest_event_id = node.id
            else:
                latest = self._nodes[self._latest_event_id]
                if node.timestamp > latest.timestamp:
                    self._latest_event_id = node.id
        else:
            assert entity_key is not None
            self._entity_index[entity_key] = node.id

        return node

    def add_event(
        self,
        content: str,
        *,
        timestamp: float,
        embedding: np.ndarray | Iterable[float] | None = None,
        attributes: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> MemoryNode:
        """Create an event node without linking it into the temporal chain."""
        node = MemoryNode(
            id=id or self._next_event_id(),
            kind=NodeKind.EVENT,
            content=content,
            timestamp=timestamp,
            embedding=embedding,
            attributes=dict(attributes or {}),
        )
        return self.add_node(node)

    def append_event(
        self,
        content: str,
        *,
        timestamp: float,
        embedding: np.ndarray | Iterable[float] | None = None,
        attributes: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> MemoryNode:
        """Create an event and append it to the temporal chain.

        The strict increase check happens before the node is inserted, so an
        out-of-order event leaves the graph unchanged.
        """
        latest = self.latest_event
        timestamp = float(timestamp)

        if latest is not None and timestamp <= latest.timestamp:
            raise TemporalOrderError(
                f"new event timestamp {timestamp!r} must be greater than the "
                f"latest event {latest.id!r} at {latest.timestamp!r}; "
                "temporal edges must point forward"
            )

        node = self.add_event(
            content,
            timestamp=timestamp,
            embedding=embedding,
            attributes=attributes,
            id=id,
        )
        if latest is not None:
            self.link_temporal(latest.id, node.id)
        return node

    def add_entity(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> MemoryNode:
        """Create or reuse an entity node by normalized name."""
        entity_key = _normalize_name(name)
        existing = self._entity_index.get(entity_key)
        if existing is not None:
            if id is not None and id != existing:
                raise DuplicateNodeError(
                    f"entity {name!r} already exists as {existing!r}, not {id!r}"
                )
            return self._nodes[existing]

        node = MemoryNode(
            id=id or self._next_entity_id(),
            kind=NodeKind.ENTITY,
            content=name.strip(),
            attributes=dict(attributes or {}),
        )
        return self.add_node(node)

    def get(self, node_id: str) -> MemoryNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise NodeNotFoundError(f"unknown node {node_id!r}")
        return node

    def get_entity(self, name: str) -> MemoryNode | None:
        node_id = self._entity_index.get(_normalize_name(name))
        return self._nodes[node_id] if node_id is not None else None

    @property
    def latest_event(self) -> MemoryNode | None:
        if self._latest_event_id is None:
            return None
        return self._nodes[self._latest_event_id]

    @property
    def nx_graph(self) -> nx.MultiDiGraph:
        """Read-only view of the underlying NetworkX graph for inspection."""
        return self._graph

    def events(self) -> list[MemoryNode]:
        return sorted(
            (node for node in self._nodes.values() if node.kind is NodeKind.EVENT),
            key=lambda node: (node.timestamp, node.id),
        )

    def entities(self) -> list[MemoryNode]:
        return sorted(
            (node for node in self._nodes.values() if node.kind is NodeKind.ENTITY),
            key=lambda node: (node.content.casefold(), node.id),
        )

    def nodes(self, kind: NodeKind | str | None = None) -> list[MemoryNode]:
        if kind is None:
            return [*self.events(), *self.entities()]
        kind = NodeKind(kind)
        return self.events() if kind is NodeKind.EVENT else self.entities()

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and node_id in self._nodes

    def __iter__(self) -> Iterator[MemoryNode]:
        return iter(self._nodes.values())

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"MemoryGraph(events={stats['nodes']['event']}, "
            f"entities={stats['nodes']['entity']}, "
            f"edges={stats['edges']['total']})"
        )

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def link_temporal(
        self, source: str, target: str, *, weight: float = 1.0
    ) -> MemoryEdge:
        """Add a forward-in-time edge from one event to another."""
        source_node = self._require_event(source)
        target_node = self._require_event(target)

        if source_node.id == target_node.id:
            raise ValueError("a temporal edge cannot connect an event to itself")
        if source_node.timestamp >= target_node.timestamp:
            raise TemporalOrderError(
                f"temporal edge {source!r} -> {target!r} must point forward in "
                f"time ({source_node.timestamp!r} >= {target_node.timestamp!r})"
            )

        return self._add_edge(
            source,
            target,
            EdgeType.TEMPORAL,
            weight,
            {"delta_seconds": target_node.timestamp - source_node.timestamp},
        )

    def link_semantic(
        self, first: str, second: str, similarity: float
    ) -> tuple[MemoryEdge, MemoryEdge]:
        """Add symmetric similarity edges between two events."""
        first_node = self._require_event(first)
        second_node = self._require_event(second)

        if first_node.id == second_node.id:
            raise ValueError("a semantic edge cannot connect an event to itself")

        attributes = {"similarity": float(similarity)}
        forward = self._add_edge(
            first, second, EdgeType.SEMANTIC, similarity, attributes
        )
        backward = self._add_edge(
            second, first, EdgeType.SEMANTIC, similarity, attributes
        )
        return forward, backward

    def link_causal(
        self,
        cause: str,
        effect: str,
        confidence: float = 1.0,
        *,
        evidence: str | None = None,
    ) -> MemoryEdge:
        """Add a directed cause -> effect edge."""
        cause_node = self._require_event(cause)
        effect_node = self._require_event(effect)

        if cause_node.id == effect_node.id:
            raise ValueError("a causal edge cannot connect an event to itself")

        attributes: dict[str, Any] = {}
        if evidence is not None:
            attributes["evidence"] = evidence
        return self._add_edge(
            cause, effect, EdgeType.CAUSAL, confidence, attributes
        )

    def link_entity(
        self, event: str, entity: str
    ) -> tuple[MemoryEdge, MemoryEdge]:
        """Link an event to an entity node in both directions."""
        event_node = self._require(event)
        entity_node = self._require(entity)

        if event_node.kind is NodeKind.ENTITY and entity_node.kind is NodeKind.EVENT:
            event_node, entity_node = entity_node, event_node

        if (
            event_node.kind is not NodeKind.EVENT
            or entity_node.kind is not NodeKind.ENTITY
        ):
            raise ValueError(
                "entity links require exactly one event node and one entity node"
            )

        attributes = {"name": entity_node.content}
        forward = self._add_edge(
            event_node.id, entity_node.id, EdgeType.ENTITY, 1.0, attributes
        )
        backward = self._add_edge(
            entity_node.id, event_node.id, EdgeType.ENTITY, 1.0, attributes
        )
        return forward, backward

    def _add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float,
        attributes: Mapping[str, Any] | None = None,
    ) -> MemoryEdge:
        edge_type = EdgeType(edge_type)
        weight = self._validate_weight(edge_type, weight)
        payload = dict(attributes or {})

        self._graph.add_edge(
            source,
            target,
            key=edge_type.value,
            type=edge_type,
            weight=weight,
            attributes=payload,
        )
        return MemoryEdge(source, target, edge_type, weight, payload)

    @staticmethod
    def _validate_weight(edge_type: EdgeType, weight: float) -> float:
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{edge_type.value} edge weight must be a number"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"{edge_type.value} edge weight must be finite")
        if value <= 0.0 or value > 1.0:
            raise ValueError(
                f"{edge_type.value} edge weight must be in (0, 1], got {value!r}"
            )
        return value

    # ------------------------------------------------------------------
    # Edge access
    # ------------------------------------------------------------------

    def out_edges(
        self,
        node_id: str,
        edge_types: EdgeType | str | Iterable[EdgeType | str] | None = None,
    ) -> list[MemoryEdge]:
        """Return outgoing edges, strongest first, optionally filtered by type."""
        self._require(node_id)
        allowed = _normalize_edge_types(edge_types)

        edges: list[MemoryEdge] = []
        for _, target, _, data in self._graph.out_edges(
            node_id, keys=True, data=True
        ):
            edge_type = EdgeType(data["type"])
            if allowed is not None and edge_type not in allowed:
                continue
            edges.append(
                MemoryEdge(
                    source=node_id,
                    target=target,
                    type=edge_type,
                    weight=float(data["weight"]),
                    attributes=dict(data.get("attributes", {})),
                )
            )

        edges.sort(key=lambda edge: (-edge.weight, edge.type.value, edge.target))
        return edges

    def neighbors(
        self,
        node_id: str,
        edge_types: EdgeType | str | Iterable[EdgeType | str] | None = None,
    ) -> list[MemoryNode]:
        """Return distinct adjacent nodes in edge-priority order."""
        seen: set[str] = set()
        result: list[MemoryNode] = []
        for edge in self.out_edges(node_id, edge_types):
            if edge.target in seen:
                continue
            seen.add(edge.target)
            result.append(self._nodes[edge.target])
        return result

    def has_edge(
        self, source: str, target: str, edge_type: EdgeType | str
    ) -> bool:
        edge_type = EdgeType(edge_type)
        return self._graph.has_edge(source, target, key=edge_type.value)

    def edge(
        self, source: str, target: str, edge_type: EdgeType | str
    ) -> MemoryEdge | None:
        edge_type = EdgeType(edge_type)
        if not self._graph.has_edge(source, target, key=edge_type.value):
            return None
        data = self._graph.get_edge_data(source, target, key=edge_type.value)
        return MemoryEdge(
            source=source,
            target=target,
            type=edge_type,
            weight=float(data["weight"]),
            attributes=dict(data.get("attributes", {})),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        edge_counts = {edge_type.value: 0 for edge_type in EdgeType}
        for _, _, data in self._graph.edges(data=True):
            edge_counts[EdgeType(data["type"]).value] += 1

        event_count = sum(
            1 for node in self._nodes.values() if node.kind is NodeKind.EVENT
        )
        entity_count = len(self._nodes) - event_count

        return {
            "nodes": {
                "event": event_count,
                "entity": entity_count,
                "total": len(self._nodes),
            },
            "edges": {
                **edge_counts,
                "total": sum(edge_counts.values()),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, node_id: str) -> MemoryNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise NodeNotFoundError(f"unknown node {node_id!r}")
        return node

    def _require_event(self, node_id: str) -> MemoryNode:
        node = self._require(node_id)
        if node.kind is not NodeKind.EVENT:
            raise ValueError(f"node {node_id!r} is not an event")
        return node

    def _next_event_id(self) -> str:
        while True:
            self._event_sequence += 1
            candidate = f"evt_{self._event_sequence:04d}"
            if candidate not in self._nodes:
                return candidate

    def _next_entity_id(self) -> str:
        while True:
            self._entity_sequence += 1
            candidate = f"ent_{self._entity_sequence:04d}"
            if candidate not in self._nodes:
                return candidate


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("entity name must be a string")
    key = " ".join(name.split()).casefold()
    if not key:
        raise ValueError("entity name must not be empty")
    return key


def _normalize_edge_types(
    edge_types: EdgeType | str | Iterable[EdgeType | str] | None,
) -> set[EdgeType] | None:
    if edge_types is None:
        return None
    if isinstance(edge_types, str):
        return {EdgeType(edge_types)}
    return {EdgeType(edge_type) for edge_type in edge_types}
