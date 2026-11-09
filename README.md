# MAGMA: Multi-Graph Agentic Memory

An independent implementation of **MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents** (Jiang et al., ACL 2026).

[![Paper](https://img.shields.io/badge/arXiv-2601.03236-b31b1b)](https://arxiv.org/abs/2601.03236)
[![Conference](https://img.shields.io/badge/ACL-2026-blue)](https://aclanthology.org/2026.acl-long.1709/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

MAGMA is a memory architecture for long-horizon LLM agents. Rather than storing interaction history as a flat vector index, it represents each memory item as an **event node** and organizes the same events across four orthogonal relational graphs:

- **Semantic** — conceptually similar events
- **Temporal** — an immutable chronological backbone
- **Causal** — directed cause → effect links
- **Entity** — events connected through people, places, and objects

Retrieval is formulated as intent-conditioned traversal over these views. A router classifies the query intent (`WHY`, `WHEN`, or `ENTITY`), and a beam search expands the most relevant subgraph while biasing edge selection toward the detected intent. Retrieved nodes are linearized into a provenance-preserving context (`<t:timestamp> content <ref:id>`) before being passed to the answer model.

The write path is split into two streams:

- **Fast path** — latency-sensitive ingestion: segmentation, embedding, vector/keyword indexing, and temporal linking.
- **Slow path** — asynchronous consolidation: inspects the local neighborhood and infers causal and entity structure.

This separation keeps ingestion responsive while the memory graph becomes progressively denser and more useful for multi-hop reasoning.

## Architecture

```
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

## Planned layout

```
magma/
├── magma/
│   ├── graph.py        # Event nodes, typed edges, graph store
│   ├── embed.py        # Embedding backend (MiniLM default)
│   ├── llm.py          # OpenAI-compatible LLM providers
│   ├── extract.py      # Structured event extraction
│   ├── ingest.py       # Fast path + consolidation pipeline
│   ├── retrieve.py     # Intent routing, RRF, beam traversal
│   ├── answer.py       # Context linearization + QA
│   └── config.py       # Configuration and hyperparameters
├── data/               # Conversations and evaluation sets
├── tests/              # Unit and integration tests
├── demo.py             # End-to-end demo
├── eval_locomo.py      # LoCoMo evaluation harness
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Event-node graph with semantic, temporal, causal, and entity edges
- [ ] Fast-path ingestion with vector and keyword indexes
- [ ] LLM-based consolidation for causal and entity links
- [ ] Intent-aware routing and temporal normalization
- [ ] RRF anchor fusion
- [ ] Adaptive beam-search traversal
- [ ] Provenance-preserving context linearization
- [ ] End-to-end QA pipeline
- [ ] Synthetic multi-session demo
- [ ] LoCoMo evaluation
- [ ] Ablation harness (no causal / no temporal / no entity / no adaptive policy)

## Quickstart

> The core modules are under active development. This section will be updated as they land.

```bash
git clone <repository-url>
cd magma

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY or NVIDIA_API_KEY to .env
python demo.py
```

## Configuration

| Component | Default |
|---|---|
| LLM providers | Groq, NVIDIA NIM, or any OpenAI-compatible endpoint |
| Embeddings | `all-MiniLM-L6-v2` (384-d, CPU-friendly) |
| Graph store | NetworkX `MultiDiGraph` |
| Vector index | NumPy cosine similarity |
| RRF constant | `k = 60` |
| Retrieval depth | up to 5 hops |
| Traversal budget | up to 200 nodes |

All parameters are exposed through `magma/config.py` so retrieval behavior can be ablated and reproduced.

## Design notes

- **Stable graph, adaptive retrieval.** Causal and entity structure is inferred at consolidation time and stored independently of any query. Query intent influences traversal priority, not the stored graph.
- **Provenance by construction.** Every linearized node carries its timestamp and reference ID, which makes retrieved evidence auditable.
- **Pluggable backends.** The LLM and embedding layers sit behind small interfaces, so local and hosted models can be swapped without touching the memory logic.
- **No dataset-specific heuristics.** The implementation targets the paper's algorithm rather than tuning to a particular benchmark.

## Evaluation

The implementation targets the two benchmarks used in the paper:

- **LoCoMo** — long multi-session conversations with single-hop, multi-hop, temporal, open-domain, and adversarial questions.
- **LongMemEval** — ultra-long interaction histories with information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and preference tracking.

Metrics: LLM-as-a-judge, token-level F1, and BLEU-1, plus retrieval-side diagnostics (latency, tokens per query, and retrieved-node counts).

## Citation

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

## License

MIT. See [LICENSE](LICENSE).

This is an independent implementation of the architecture described in the paper and is not affiliated with or endorsed by the original authors.
