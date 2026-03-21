# GSEM: Graph-based Self-Evolving Memory

GSEM is a graph-structured experience memory framework for AI agents. It automatically extracts reusable experiences from task trajectories, organizes them into a structured experience graph, and supports continuous online evolution.

## Directory Structure

```
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

**Phase 1 — Experience Extraction**

Runs a multi-stage pipeline (rollout → normalization → deduplication → ERV) to extract structured experiences from agent trajectories.

```bash
python main_phase1.py
```

**Phase 2 — Graph Construction**

Builds an experience graph: extracts entities, computes semantic/task similarity, and exports to NetworkX format.

```bash
python main_phase2.py
```

**Phase 3 — Online Evolution**

Runs the online TTL pipeline: the agent retrieves relevant experiences from the graph, solves new cases, and incrementally updates the graph.

```bash
python main_phase3.py
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```
