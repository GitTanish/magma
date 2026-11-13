"""MAGMA: a multi-graph agentic memory architecture."""

from .graph import (
    DuplicateNodeError,
    EdgeType,
    MemoryEdge,
    MemoryGraph,
    MemoryNode,
    NodeKind,
    NodeNotFoundError,
    TemporalOrderError,
)

__all__ = [
    "DuplicateNodeError",
    "EdgeType",
    "MemoryEdge",
    "MemoryGraph",
    "MemoryNode",
    "NodeKind",
    "NodeNotFoundError",
    "TemporalOrderError",
]

__version__ = "0.1.0"
