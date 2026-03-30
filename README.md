# GSEM: Graph-based Self-Evolving Memory for Experience-Augmented Clinical Reasoning

GSEM is a graph-based experience memory framework for clinical reasoning agents. It extracts reusable experiences from reasoning trajectories, organizes them into a dual-layer memory graph, retrieves applicable experiences for new cases, and continuously calibrates memory quality and inter-experience relations through online feedback.

<p align="center">
  <img src="figure/gsem_overview.pdf" alt="Overview of GSEM" width="100%">
</p>

## Overview

Large language model agents can benefit from reusing prior decision experience, but flat memory banks often store experiences as isolated records. This makes it difficult to:

- verify whether a retrieved experience is truly applicable under the current clinical conditions;
- model how multiple experiences should be jointly used;
- continuously refine memory reliability after deployment.

GSEM addresses these challenges with a three-stage framework:

1. **Memory Construction**: extract structured experiences from successful and failed reasoning trajectories, validate their initial reliability, and build a dual-layer memory graph.
2. **Memory Retrieval**: perform hybrid seed recall and graph-based multi-seed traversal to retrieve boundary-aware, composition-compatible experiences.
3. **Memory Evolution**: update node quality and edge weights using task feedback, enabling the memory graph to self-evolve over time.

## Key Ideas

- **Dual-layer memory graph**
  - **Entity layer** models the internal decision structure of each experience.
  - **Experience layer** models relations across experiences.
- **Experience types**
  - **Indication**: reusable successful decision knowledge.
  - **Contraindication**: reusable failure-derived knowledge that highlights what should be avoided.
- **Applicability-aware retrieval**
  - combines entity-based recall, embedding-based recall, reranking, and graph traversal.
- **Online self-evolution**
  - calibrates experience quality and relation weights without rewriting experience content.

## Repository Structure

```text
GSEM/
├── main_phase1.py          # Phase 1 entry: experience extraction
├── main_phase2.py          # Phase 2 entry: graph construction
├── main_phase3.py          # Phase 3 entry: online evolution
├── experiences.jsonl       # Extracted experience data
├── requirements.txt
├── .env.example
├── src/
│   ├── shared/             # Shared modules (config, logger, utils)
│   ├── phase1/             # Phase 1: experience extraction pipeline
│   │   ├── pipeline.py
│   │   ├── prompts.py
│   │   ├── prompt_provider.py
│   │   ├── stages/         # Rollout, normalization, deduplication, ERV, etc.
│   │   └── agents/         # ReAct agent
│   ├── phase2/             # Phase 2: graph construction
│   │   └── graph/          # Entity extraction, normalization, similarity scoring, export
│   └── phase3/             # Phase 3: online evolution
│       ├── ttl/            # Online evolution pipeline and reasoning agent
│       └── retrieval/      # Graph-based experience retrieval
├── data/                   # Intermediate and processed data
└── evaluation/
    └── medrb/
        └── data/           # Evaluation test splits
```

## Three Phases

### Phase 1 — Experience Extraction

Runs a multi-stage pipeline (rollout → normalization → deduplication → ERV) to extract structured experiences from agent trajectories.

```bash
python main_phase1.py
```

### Phase 2 — Graph Construction

Builds the dual-layer memory graph by extracting entities, computing similarity signals, and exporting the graph structure.

```bash
python main_phase2.py
```

### Phase 3 — Online Evolution

Runs the online TTL pipeline: the agent retrieves relevant experiences from the graph, solves new cases, and incrementally updates memory quality and relation weights.

```bash
python main_phase3.py
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

## Method Summary

### Phase 1: Memory Construction

- Sample multiple reasoning trajectories for each case.
- Distill successful trajectories into **Indication** experiences.
- Distill failure-success divergences into **Contraindication** experiences.
- Run **Experience Reliability Validation (ERV)** to initialize experience quality.
- Construct the dual-layer memory graph.

### Phase 2: Memory Retrieval

- Use **entity-based recall** to match decision-relevant conditions.
- Use **embedding-based recall** to capture semantic similarity.
- Merge and rerank retrieved candidates.
- Start from multiple seeds and perform graph traversal to collect compatible experiences.

### Phase 3: Memory Evolution

- Use task feedback after each case.
- Update node quality scores for activated experiences.
- Update edge weights for co-activated experience pairs.
- Optionally insert newly extracted experiences into the graph.

## Paper

**Title:** *GSEM: Graph-based Self-Evolving Memory for Experience-Augmented Clinical Reasoning*

If you use this repository, please cite the corresponding paper.

```bibtex
@article{han2026gsem,
  title={GSEM: Graph-based Self-Evolving Memory for Experience-Augmented Clinical Reasoning},
  author={Han, Xiao and Fan, Yuzheng and Zhao, Sendong and Wang, Haochun and Qin, Bing},
  journal={arXiv preprint arXiv},
  year={2026}
}
```

## Notes

- This repository currently focuses on the code framework for the three stages of GSEM.
- Dataset preparation, environment variables, and model backends should be configured according to your local setup.
- The framework is designed for research use.