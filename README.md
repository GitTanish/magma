# MAGMA

**A multi-graph memory architecture for long-horizon agents.**

Independent implementation of *MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents* (Jiang et al., ACL 2026).

**Status:** graph store implemented. Retrieval pipeline planned.

[![Paper](https://img.shields.io/badge/arXiv-2601.03236-b31b1b)](https://arxiv.org/abs/2601.03236)
[![Conference](https://img.shields.io/badge/ACL-2026-blue)](https://aclanthology.org/2026.acl-long.1709/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 0. Reading contract

This README is intentionally literal.

It uses exactly three status words:

| Status | Meaning |
|---|---|
| **Implemented** | Code exists in this repository and is covered by tests. |
| **Planned** | No code exists. The interface may change. |
| **Out of scope** | Deliberately not built. |

A sentence without a status word describes the target architecture. It does not claim that code exists.

Ambiguity is treated as a defect.

## 1. Definition

MAGMA is a memory architecture for long-horizon LLM agents.

It replaces a single flat vector index with:

- one set of memory nodes;
- four typed relations over that set;
- a retrieval policy that selects relations based on query intent.

The central claim:

> The stored graph is query-independent. The retrieval policy is query-dependent.

## 2. Data model

There is one node set. There are two node kinds.

```text
Event  = (id, content, timestamp, embedding, attributes)
Entity = (id, name, attributes)
```

| Field | Event | Entity | Constraint |
|---|---|---|---|
| `id` | required | required | unique, non-empty string |
| `content` | required | required | event text or entity name |
| `timestamp` | required | optional | finite numeric value |
| `embedding` | optional | optional | finite, non-empty 1-D float vector |
| `attributes` | optional | optional | caller-defined dictionary |

Events are observations. Entities are people, places, objects, or other things an event can mention.

## 3. Relations

Four relations share the same node set.

| Relation | Logical direction | Storage | Weight |
|---|---|---|---|
| **Semantic** | symmetric | two directed edges | cosine similarity |
| **Temporal** | earlier → later | one directed edge | bounded score, default `1.0` |
| **Causal** | cause → effect | one directed edge | confidence |
| **Entity** | event ↔ entity | two directed edges | `1.0` |

Causal direction is semantic. It does not have to match temporal direction.

Weights are bounded scores in `(0, 1]`. They are not probabilities.

## 4. Invariants

These are enforced by `magma/graph.py`.

1. Node ids are unique.
2. Event nodes require a finite timestamp.
3. A temporal edge requires `source.timestamp < target.timestamp`.
4. `append_event` requires a timestamp greater than the latest event.
5. A rejected write does not mutate the graph.
6. Semantic edges require two distinct event nodes.
7. Causal edges require two distinct event nodes.
8. Entity links require exactly one event node and one entity node.
9. Edge weights are finite and in `(0, 1]`.
10. Entity names are deduplicated by casefolded, whitespace-collapsed exact match.
11. Alias resolution is not implemented. `Melanie`, `Mel`, and `my sister` are three entities.
12. The graph is in memory. Persistence is not implemented.
13. There is no edge deletion API. Semantic rebuilds are additive.

## 5. Architecture

```text
                         ┌──────────────────────┐
                         │        Query         │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Intent router       │
                         │  WHY / WHEN / ENTITY │
                         │  + temporal parser   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Anchor retrieval    │
                         │  vector + keyword +  │
                         │  time, fused via RRF │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Adaptive traversal  │
                         │  over semantic /     │
                         │  temporal / causal / │
                         │  entity graphs       │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Structure-aware     │
                         │  context linearizer  │
                         └──────────┬───────────┘
                                    │
                            ┌───────▼────────┐
                            │  Answer model  │
                            └────────────────┘
```

Each arrow is a data dependency. It is not a process.

## 6. Write path

The write path has two streams. They have different latency budgets.

### Fast path

Latency-sensitive. Must not block on an LLM.

| Step | Action | Status |
|---|---|---|
| 1 | Receive an event | Planned |
| 2 | Segment the event | Planned |
| 3 | Embed the event | Planned |
| 4 | Append the event node | **Implemented** |
| 5 | Add the temporal edge from the previous event | **Implemented** |
| 6 | Update vector and keyword indexes | Planned |

### Slow path

Consolidation. Runs after ingestion.

| Step | Action | Status |
|---|---|---|
| 1 | Inspect the event's 2-hop neighborhood | Planned |
| 2 | Infer causal candidates with an LLM | Planned |
| 3 | Infer entity mentions with an LLM | Planned |
| 4 | Validate candidates | Planned |
| 5 | Write causal and entity edges with provenance | Planned |

A query issued after ingestion may see a partially consolidated graph. `consolidate_now()` is planned for tests and demos.

## 7. Read path

All steps are **Planned**.

1. Receive the query.
2. Classify intent as `WHY`, `WHEN`, or `ENTITY`.
3. Parse relative time expressions into absolute intervals.
4. Retrieve anchors with vector search, keyword search, and time filters.
5. Fuse ranked lists with Reciprocal Rank Fusion, `k = 60`.
6. Expand a beam over the four relations.
7. Score transitions with the intent-conditioned structural prior.
8. Linearize retrieved nodes as `<t:timestamp> content <ref:id>`.
9. Answer using only the linearized context.

The traversal score follows the paper:

```text
S(n_j | n_i, q) = exp(λ₁ · φ(type(e_ij), T_q) + λ₂ · sim(n_j, q))
score_v         = score_u · γ + S(n_j | n_i, q)
```

Table 6 weights are not used raw. Raw weights between `2.5` and `6.0` inside `exp` are numerically hostile. Calibration is a retrieval-layer concern.

## 8. Status

| Component | Status | Evidence |
|---|---|---|
| Event and entity nodes | **Implemented** | `magma/graph.py` |
| Semantic, temporal, causal, and entity edges | **Implemented** | `magma/graph.py` |
| Temporal order validation | **Implemented** | `tests/test_graph.py` |
| Entity-name deduplication | **Implemented** | `tests/test_graph.py` |
| Weight validation | **Implemented** | `tests/test_graph.py` |
| Embedding backends | Planned | none |
| Semantic edge construction | Planned | none |
| Fast-path ingestion | Planned | none |
| Vector and keyword indexes | Planned | none |
| LLM consolidation | Planned | none |
| Intent router | Planned | none |
| Temporal normalization | Planned | none |
| RRF anchor fusion | Planned | none |
| Beam-search traversal | Planned | none |
| Context linearization | Planned | none |
| End-to-end QA | Planned | none |
| Synthetic demo | Planned | none |
| LoCoMo evaluation | Planned | none |
| Ablation harness | Planned | none |

Current test count: **14 passing**.

## 9. Repository layout

Present:

```text
magma/
├── magma/
│   ├── __init__.py
│   └── graph.py
├── tests/
│   └── test_graph.py
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Planned:

```text
magma/
├── magma/
│   ├── embed.py        # embedding backends
│   ├── semantic.py     # semantic edge construction
│   ├── llm.py          # OpenAI-compatible providers
│   ├── extract.py      # structured event extraction
│   ├── ingest.py       # fast path and consolidation
│   ├── retrieve.py     # router, RRF, beam search
│   ├── answer.py       # linearization and QA
│   └── config.py       # configuration
├── data/               # conversations and evaluation sets
├── demo.py             # end-to-end demo
└── eval_locomo.py      # LoCoMo harness
```

## 10. Quickstart

The graph store is the only runnable component.

```bash
git clone https://github.com/GitTanish/magma.git
cd magma

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q
```

Expected result:

```text
14 passed
```

There is no demo yet. `demo.py` is planned. The test suite is the only executable verification.

## 11. Defaults

These are intended defaults. No `config.py` exists yet.

| Component | Default | Status |
|---|---|---|
| LLM providers | Groq, NVIDIA NIM, or any OpenAI-compatible endpoint | Planned |
| Embeddings | `all-MiniLM-L6-v2`, 384 dimensions, CPU-friendly | Planned |
| Graph store | NetworkX `MultiDiGraph` | **Implemented** |
| Vector index | NumPy cosine similarity | Planned |
| RRF constant | `k = 60` | Planned |
| Retrieval depth | up to 5 hops | Planned |
| Traversal budget | up to 200 nodes | Planned |

## 12. Design rules

1. **Stable graph, adaptive retrieval.** Causal and entity structure is inferred during consolidation. It is stored independently of any query.
2. **Provenance by construction.** Every linearized node carries a timestamp and reference id.
3. **Validation before mutation.** Illegal writes fail before state changes.
4. **Pluggable backends.** LLM and embedding layers sit behind small interfaces.
5. **No dataset-specific heuristics.** No hardcoded month maps. No per-benchmark entity dictionaries. No regex query classifiers tuned to one dataset.

Rule 5 is a non-goal, not an aspiration.

## 13. Known limitations

1. Entity aliases are unresolved.
2. The graph has no deletion API.
3. Semantic edge construction is not implemented.
4. Consolidation is not implemented.
5. Retrieval is not implemented.
6. Answer generation is not implemented.
7. Nothing is persisted.
8. There is no benchmark result to report.

## 14. Evaluation plan

Planned benchmarks:

- **LoCoMo** — long multi-session conversations with single-hop, multi-hop, temporal, open-domain, and adversarial questions.
- **LongMemEval** — long interaction histories with information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and preference tracking.

Planned metrics:

- LLM-as-a-judge.
- Token-level F1.
- BLEU-1.
- Retrieval latency.
- Tokens per query.
- Retrieved-node count.

No evaluation has been run. No number in this README is a result.

## 15. Citation

```bibtex
@inproceedings{jiang-etal-2026-magma,
  title     = "{MAGMA}: A Multi-Graph based Agentic Memory Architecture for {AI} Agents",
  author    = "Jiang, Dongming and Li, Yi and Li, Guanpeng and Li, Bingzhe",
  editor    = "Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David",
  booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month     = jul,
  year      = "2026",
  address   = "San Diego, California, United States",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2026.acl-long.1709/",
  doi       = "10.18653/v1/2026.acl-long.1709",
  pages     = "36848--36865"
}
```

## 16. License

MIT. See [LICENSE](LICENSE).

This is an independent implementation of the architecture described in the paper. It is not affiliated with or endorsed by the original authors.
