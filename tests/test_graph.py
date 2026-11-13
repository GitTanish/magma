"""Unit tests for the typed memory graph.

These tests encode the invariants the rest of the pipeline depends on:

* events carry time, content, embeddings, and attributes;
* the temporal backbone only moves forward and links consecutive events;
* semantic similarity is symmetric;
* causation is directional;
* entity mentions share one entity node instead of duplicating text.
"""

from __future__ import annotations

import numpy as np
import pytest

from magma import (
    DuplicateNodeError,
    EdgeType,
    MemoryGraph,
    MemoryNode,
    NodeKind,
    TemporalOrderError,
)


def test_events_are_nodes_with_time_embeddings_and_attributes():
    graph = MemoryGraph()
    event = graph.add_event(
        "I moved to Kyoto in April.",
        timestamp=1_700_000_000,
        embedding=np.array([1.0, 0.0, 0.0]),
        attributes={"session": 1},
    )

    assert event.kind is NodeKind.EVENT
    assert graph.get(event.id) is event
    assert event.embedding.shape == (3,)
    assert event.attributes["session"] == 1


def test_event_node_requires_timestamp():
    with pytest.raises(ValueError, match="timestamp"):
        MemoryNode(id="e1", kind=NodeKind.EVENT, content="no time")


def test_append_event_builds_a_consecutive_temporal_chain():
    graph = MemoryGraph()
    first = graph.append_event("I landed in Osaka.", timestamp=100)
    second = graph.append_event("I took the train to Kyoto.", timestamp=200)
    third = graph.append_event("I visited Fushimi Inari.", timestamp=300)

    assert graph.has_edge(first.id, second.id, EdgeType.TEMPORAL)
    assert graph.has_edge(second.id, third.id, EdgeType.TEMPORAL)
    assert not graph.has_edge(first.id, third.id, EdgeType.TEMPORAL)
    assert [
        edge.target for edge in graph.out_edges(first.id, EdgeType.TEMPORAL)
    ] == [second.id]


def test_temporal_edge_cannot_point_backwards():
    graph = MemoryGraph()
    graph.append_event("The trip ended.", timestamp=200)

    with pytest.raises(TemporalOrderError, match="forward"):
        graph.append_event("The trip began.", timestamp=100)

    assert len(graph) == 1


def test_explicit_temporal_link_validates_order():
    graph = MemoryGraph()
    later = graph.add_event("later", timestamp=20)
    earlier = graph.add_event("earlier", timestamp=10)

    with pytest.raises(TemporalOrderError):
        graph.link_temporal(later.id, earlier.id)

    graph.link_temporal(earlier.id, later.id)
    assert graph.has_edge(earlier.id, later.id, EdgeType.TEMPORAL)


def test_semantic_links_are_bidirectional_and_weighted():
    graph = MemoryGraph()
    first = graph.add_event("I love hiking.", timestamp=1)
    second = graph.add_event("Trail running is my hobby.", timestamp=2)

    graph.link_semantic(first.id, second.id, similarity=0.81)

    forward = graph.edge(first.id, second.id, EdgeType.SEMANTIC)
    backward = graph.edge(second.id, first.id, EdgeType.SEMANTIC)
    assert forward is not None and backward is not None
    assert forward.weight == pytest.approx(0.81)
    assert backward.weight == pytest.approx(0.81)


def test_causal_links_are_directional():
    graph = MemoryGraph()
    cause = graph.add_event("My flight was cancelled.", timestamp=1)
    effect = graph.add_event("I missed the wedding.", timestamp=2)

    graph.link_causal(
        cause.id,
        effect.id,
        confidence=0.7,
        evidence="the cancellation explains the missed wedding",
    )

    assert graph.has_edge(cause.id, effect.id, EdgeType.CAUSAL)
    assert not graph.has_edge(effect.id, cause.id, EdgeType.CAUSAL)
    assert (
        graph.edge(cause.id, effect.id, EdgeType.CAUSAL).attributes["evidence"]
        == "the cancellation explains the missed wedding"
    )


def test_entity_nodes_connect_event_mentions():
    graph = MemoryGraph()
    event = graph.append_event("Melanie plays the violin.", timestamp=1)
    entity = graph.add_entity("Melanie")
    graph.link_entity(event.id, entity.id)

    assert entity.kind is NodeKind.ENTITY
    assert graph.has_edge(event.id, entity.id, EdgeType.ENTITY)
    assert graph.has_edge(entity.id, event.id, EdgeType.ENTITY)
    assert [node.id for node in graph.neighbors(entity.id, EdgeType.ENTITY)] == [
        event.id
    ]


def test_entity_names_are_deduplicated_by_normalized_form():
    graph = MemoryGraph()
    first = graph.add_entity("Melanie")
    second = graph.add_entity("  melanie ")

    assert first is second
    assert graph.get_entity("MELANIE") is first
    assert len(graph.entities()) == 1


def test_neighbor_queries_filter_by_edge_type():
    graph = MemoryGraph()
    anchor = graph.add_event("A", timestamp=1)
    semantic = graph.add_event("B", timestamp=2)
    causal = graph.add_event("C", timestamp=3)
    graph.link_semantic(anchor.id, semantic.id, 0.9)
    graph.link_causal(anchor.id, causal.id, 0.8)
    graph.link_temporal(anchor.id, semantic.id)

    semantic_neighbors = graph.neighbors(anchor.id, EdgeType.SEMANTIC)
    causal_neighbors = graph.neighbors(anchor.id, EdgeType.CAUSAL)
    both_neighbors = graph.neighbors(
        anchor.id, [EdgeType.SEMANTIC, EdgeType.CAUSAL]
    )

    assert [node.id for node in semantic_neighbors] == [semantic.id]
    assert [node.id for node in causal_neighbors] == [causal.id]
    assert {node.id for node in both_neighbors} == {semantic.id, causal.id}


def test_stats_report_nodes_and_directed_edge_counts():
    graph = MemoryGraph()
    first = graph.append_event("A", timestamp=1)
    second = graph.append_event("B", timestamp=2)
    entity = graph.add_entity("Alice")
    graph.link_entity(first.id, entity.id)
    graph.link_semantic(first.id, second.id, 0.9)
    graph.link_causal(first.id, second.id, 0.6)

    stats = graph.stats()

    assert stats["nodes"] == {"event": 2, "entity": 1, "total": 3}
    assert stats["edges"] == {
        "semantic": 2,
        "temporal": 1,
        "causal": 1,
        "entity": 2,
        "total": 6,
    }


def test_duplicate_node_ids_are_rejected():
    graph = MemoryGraph()
    graph.add_event("A", timestamp=1, id="evt")

    with pytest.raises(DuplicateNodeError):
        graph.add_event("B", timestamp=2, id="evt")


def test_edge_weights_are_bounded_probabilities():
    graph = MemoryGraph()
    first = graph.add_event("A", timestamp=1)
    second = graph.add_event("B", timestamp=2)

    with pytest.raises(ValueError, match="weight"):
        graph.link_semantic(first.id, second.id, 1.2)
    with pytest.raises(ValueError, match="weight"):
        graph.link_causal(first.id, second.id, 0.0)


def test_semantic_links_require_events():
    graph = MemoryGraph()
    event = graph.add_event("A", timestamp=1)
    entity = graph.add_entity("Alice")

    with pytest.raises(ValueError, match="event"):
        graph.link_semantic(event.id, entity.id, 0.9)
