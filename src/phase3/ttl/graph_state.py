"""TTL: Graph State Management.

Maintains the evolving experience graph and implements:
  - Step 9: Merge new TTL experience into graph (add node + edges)
  - Step 10: Online weight update (node quality + edge correction)
  - Dynamic edge materialisation: when W_effective crosses 0.35

Working-copy layout (all TTL updates go here, static data is never touched):
  data/ttl/experiences/experiences.jsonl      ← full copy (static 1104 + TTL new)
  data/ttl/experience_graph_adj/experience_graph_adj.jsonl  ← full adj copy
  data/ttl/experience_edges/experience_edges.jsonl          ← full edges copy

Dynamic-only state (qualities, corrections, counters):
  data/ttl/graph_state.json

Update formulas (TTL technical doc):
  delta_t = 2*score_t - 1
  a_i     = (1/log(2+rank_i)) / sum_k(1/log(2+rank_k))   rank-decay credit
  b_ij    = a_i*a_j / sum_{(u,v) in P_t}(a_u*a_v)         edge credit
  eta_q   = eta_q0 / (1+n_i)^rho
  eta_w   = eta_w0 / (1+n_ij)^rho
  q_i    <- clip(q_i + eta_q*a_i*delta_t, 0, 1)
  theta_ij<- theta_ij + eta_w*b_ij*delta_t
  W_ij    = clip(W_ij^0 + theta_ij, 0, 1)
"""
import json
import math
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Static sources (READ-ONLY, never modified)
# Experience source prefers role_edges output to retain the richest fields.
STATIC_EXP_CANDIDATES = [
    os.path.join("data", "role_edges", "role_edges.jsonl"),
    os.path.join("data", "experiences", "experiences.jsonl"),
    os.path.join("data", "preprocessed", "experiences.jsonl"),
]
STATIC_ADJ_PATH   = os.path.join("data", "experience_graph_adj", "experience_graph_adj.jsonl")
STATIC_EDGES_PATH = os.path.join("data", "experience_edges", "experience_edges.jsonl")

TTL_DIR_DEFAULT = os.path.join("data", "ttl")

# Sub-paths inside ttl_dir (mirror static structure)
TTL_EXP_DIR   = "experiences"
TTL_EXP_FILE  = "experiences.jsonl"           # full copy: static + TTL
TTL_ADJ_DIR   = "experience_graph_adj"
TTL_ADJ_FILE  = "experience_graph_adj.jsonl"
TTL_EDGES_DIR = "experience_edges"
TTL_EDGES_FILE = "experience_edges.jsonl"
TTL_STATE_FILE = "graph_state.json"


class GraphState:
    """Maintains the evolving TTL experience graph across online cases."""

    def __init__(
        self,
        experiences: Dict[str, Dict],
        adjacency: Dict[str, List[Dict]],
        node_qualities: Dict[str, float],
        node_visit_counts: Dict[str, int],
        edge_corrections: Dict[str, float],
        edge_visit_counts: Dict[str, int],
        initial_weights: Dict[str, float],
        ttl_exp_counter: int = 0,
        static_exp_count: int = 0,
    ):
        self.experiences = experiences               # exp_id → exp_dict
        self.adjacency = adjacency                   # exp_id → [{neighbor, W, ...}]
        self.node_qualities = node_qualities         # exp_id → q_i
        self.node_visit_counts = node_visit_counts   # exp_id → n_i
        self.edge_corrections = edge_corrections     # edge_key → θ_ij
        self.edge_visit_counts = edge_visit_counts   # edge_key → n_ij
        self.initial_weights = initial_weights       # edge_key → W_ij^(0)
        self.ttl_exp_counter = ttl_exp_counter
        self.static_exp_count = static_exp_count

        # Pending buffers — flushed to disk in save()
        self._pending_ttl_exps: List[Dict] = []
        self._new_materialized_edges: List[Dict] = []
        self._adj_dirty: bool = False

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def edge_key(id1: str, id2: str) -> str:
        a, b = sorted([id1, id2])
        return f"{a}:{b}"

    def is_ttl(self, eid: str) -> bool:
        """True if eid belongs to a TTL-generated experience."""
        return self.experiences.get(eid, {}).get("source") == "ttl"

    def get_current_weight(self, id1: str, id2: str) -> float:
        key = self.edge_key(id1, id2)
        W0 = self.initial_weights.get(key, 0.0)
        theta = self.edge_corrections.get(key, 0.0)
        return max(0.0, min(1.0, W0 + theta))

    def next_exp_id(self) -> str:
        """Return the next sequential experience ID.

        Primary index source is the current in-memory graph (max existing exp_XXXX + 1),
        with static_exp_count/ttl_exp_counter kept as compatibility fallback. This avoids
        accidental reset to exp_0000 when a run is started with mismatched directories.
        """
        max_existing = -1
        for eid in self.experiences.keys():
            if isinstance(eid, str) and eid.startswith("exp_"):
                try:
                    max_existing = max(max_existing, int(eid.split("_", 1)[1]))
                except (ValueError, IndexError):
                    continue

        counter_based = self.static_exp_count + self.ttl_exp_counter
        next_idx = max(max_existing + 1, counter_based)

        # Keep counter monotonic relative to static_exp_count baseline.
        self.ttl_exp_counter = max(self.ttl_exp_counter + 1, next_idx - self.static_exp_count + 1)
        return f"exp_{next_idx:04d}"

    # ------------------------------------------------------------------ loaders

    @classmethod
    def _resolve_static_experience_path(cls, data_dir: str) -> str:
        candidates = [
            os.path.join(data_dir, "role_edges", "role_edges.jsonl"),
            os.path.join(data_dir, "experiences", "experiences.jsonl"),
            os.path.join(data_dir, "preprocessed", "experiences.jsonl"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        # Keep deterministic fallback even if missing; caller logs missing source.
        return candidates[0]

    @classmethod
    def _copy_static_to_ttl(cls, data_dir: str, ttl_dir: str) -> None:
        """Copy static experience / adj / edge files to TTL working dir if absent."""
        static_exp_path = cls._resolve_static_experience_path(data_dir)
        copies = [
            (static_exp_path,
             os.path.join(ttl_dir, TTL_EXP_DIR, TTL_EXP_FILE)),
            (os.path.join(data_dir, "experience_graph_adj", "experience_graph_adj.jsonl"),
             os.path.join(ttl_dir, TTL_ADJ_DIR, TTL_ADJ_FILE)),
            (os.path.join(data_dir, "experience_edges", "experience_edges.jsonl"),
             os.path.join(ttl_dir, TTL_EDGES_DIR, TTL_EDGES_FILE)),
        ]
        for src, dst in copies:
            if not os.path.exists(dst) and os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                from src.shared.logger import logger
                logger.log_info(f"  [TTL] Copied {src} → {dst}")

    @classmethod
    def _load_from_ttl_dir(cls, ttl_dir: str):
        """Load experiences and adjacency from TTL working copies."""
        experiences: Dict[str, Dict] = {}
        node_qualities: Dict[str, float] = {}
        adjacency: Dict[str, List[Dict]] = defaultdict(list)
        initial_weights: Dict[str, float] = {}

        ttl_exp_path = os.path.join(ttl_dir, TTL_EXP_DIR, TTL_EXP_FILE)
        if os.path.exists(ttl_exp_path):
            with open(ttl_exp_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        exp = json.loads(line)
                        eid = exp.get("id", "")
                        if eid:
                            experiences[eid] = exp
                            node_qualities[eid] = float(exp.get("quality", 0.5))
        else:
            from src.shared.logger import logger
            logger.log_warning(f"[TTL] Experiences file not found: {ttl_exp_path}")

        ttl_adj_path = os.path.join(ttl_dir, TTL_ADJ_DIR, TTL_ADJ_FILE)
        if os.path.exists(ttl_adj_path):
            with open(ttl_adj_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        eid = row.get("exp_id", "")
                        neighbors = row.get("neighbors", [])
                        adjacency[eid] = neighbors
                        for nb in neighbors:
                            nid = nb.get("neighbor", "")
                            W = float(nb.get("W", 0.0))
                            key = cls.edge_key(eid, nid)
                            if key not in initial_weights:
                                initial_weights[key] = W
        else:
            from src.shared.logger import logger
            logger.log_warning(f"[TTL] Adjacency file not found: {ttl_adj_path}")

        static_exp_count = sum(
            1 for e in experiences.values() if e.get("source") != "ttl"
        )
        return experiences, dict(adjacency), node_qualities, initial_weights, static_exp_count

    @classmethod
    def load_from_static(cls, data_dir: str = "data", ttl_dir: str = TTL_DIR_DEFAULT) -> "GraphState":
        """Load graph state from pre-seeded TTL working copies.

        Seeding/copying is now handled by `scripts/reset_ttl.py`.
        Runtime pipeline does not overwrite TTL working copies.
        """
        ttl_exp_path = os.path.join(ttl_dir, TTL_EXP_DIR, TTL_EXP_FILE)
        ttl_adj_path = os.path.join(ttl_dir, TTL_ADJ_DIR, TTL_ADJ_FILE)
        ttl_edges_path = os.path.join(ttl_dir, TTL_EDGES_DIR, TTL_EDGES_FILE)
        missing = [p for p in (ttl_exp_path, ttl_adj_path, ttl_edges_path) if not os.path.exists(p)]
        if missing:
            joined = "\n  - ".join(missing)
            raise FileNotFoundError(
                "TTL working copies are not ready. Please run reset first:\n"
                "  python scripts/reset_ttl.py --yes\n"
                f"Missing files:\n  - {joined}"
            )

        experiences, adjacency, node_qualities, initial_weights, static_exp_count = \
            cls._load_from_ttl_dir(ttl_dir)

        return cls(
            experiences=experiences,
            adjacency=adjacency,
            node_qualities=node_qualities,
            node_visit_counts={eid: 0 for eid in experiences},
            edge_corrections={},
            edge_visit_counts={},
            initial_weights=initial_weights,
            static_exp_count=static_exp_count,
        )

    @classmethod
    def load_or_init(
        cls,
        data_dir: str = "data",
        ttl_dir: str = TTL_DIR_DEFAULT,
    ) -> "GraphState":
        """Resume from TTL checkpoint if it exists; otherwise init from static."""
        state_path = os.path.join(ttl_dir, TTL_STATE_FILE)
        if os.path.exists(state_path):
            return cls._load_checkpoint(state_path, ttl_dir)
        return cls.load_from_static(data_dir, ttl_dir)

    @classmethod
    def _load_checkpoint(cls, state_path: str, ttl_dir: str) -> "GraphState":
        """Resume from a TTL checkpoint using TTL working copies as the source."""
        experiences, adjacency, node_qualities, initial_weights, static_exp_count = \
            cls._load_from_ttl_dir(ttl_dir)

        with open(state_path, encoding="utf-8") as f:
            ckpt = json.load(f)

        instance = cls(
            experiences=experiences,
            adjacency=adjacency,
            node_qualities=node_qualities,
            node_visit_counts={eid: 0 for eid in experiences},
            edge_corrections={},
            edge_visit_counts={},
            initial_weights=initial_weights,
            ttl_exp_counter=ckpt.get("ttl_exp_counter", 0),
            static_exp_count=ckpt.get("static_exp_count", static_exp_count),
        )

        # Overlay dynamic state from checkpoint
        instance.node_qualities.update(ckpt.get("node_qualities", {}))
        instance.node_visit_counts.update(ckpt.get("node_visit_counts", {}))
        instance.edge_corrections.update(ckpt.get("edge_corrections", {}))
        instance.edge_visit_counts.update(ckpt.get("edge_visit_counts", {}))
        return instance

    # ------------------------------------------------------------------ save

    def save(self, ttl_dir: str = TTL_DIR_DEFAULT):
        """Persist TTL state to working copies.

        Writes:
          <ttl_dir>/experiences/experiences.jsonl         ← append new TTL exps
          <ttl_dir>/experience_graph_adj/...jsonl         ← full rewrite if adj changed
          <ttl_dir>/experience_edges/...jsonl             ← append materialised edges
          <ttl_dir>/graph_state.json                      ← qualities / corrections / counters
        """
        os.makedirs(ttl_dir, exist_ok=True)

        # 1. Append new TTL experiences to full working copy
        if self._pending_ttl_exps:
            exp_path = os.path.join(ttl_dir, TTL_EXP_DIR, TTL_EXP_FILE)
            os.makedirs(os.path.dirname(exp_path), exist_ok=True)
            with open(exp_path, "a", encoding="utf-8") as f:
                for exp in self._pending_ttl_exps:
                    f.write(json.dumps(exp, ensure_ascii=False) + "\n")
            self._pending_ttl_exps.clear()

        # 2. Rewrite full adjacency file if edges were added
        if self._adj_dirty:
            adj_path = os.path.join(ttl_dir, TTL_ADJ_DIR, TTL_ADJ_FILE)
            os.makedirs(os.path.dirname(adj_path), exist_ok=True)
            with open(adj_path, "w", encoding="utf-8") as f:
                for eid in sorted(self.adjacency):
                    row = {"exp_id": eid, "neighbors": self.adjacency[eid]}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._adj_dirty = False

        # 3. Append newly materialised edges to edges file
        if self._new_materialized_edges:
            edges_path = os.path.join(ttl_dir, TTL_EDGES_DIR, TTL_EDGES_FILE)
            os.makedirs(os.path.dirname(edges_path), exist_ok=True)
            with open(edges_path, "a", encoding="utf-8") as f:
                for edge in self._new_materialized_edges:
                    f.write(json.dumps(edge, ensure_ascii=False) + "\n")
            self._new_materialized_edges.clear()

        # 4. Write lightweight graph_state.json (no adjacency — that's in the files)
        ckpt = {
            "ttl_exp_counter": self.ttl_exp_counter,
            "static_exp_count": self.static_exp_count,
            "node_qualities": self.node_qualities,
            "node_visit_counts": self.node_visit_counts,
            "edge_corrections": self.edge_corrections,
            "edge_visit_counts": self.edge_visit_counts,
        }
        state_path = os.path.join(ttl_dir, TTL_STATE_FILE)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ Step 9

    def add_experience(
        self,
        exp_id: str,
        exp: Dict[str, Any],
        scored_pairs: List[Dict[str, Any]],
        edge_threshold: float = 0.35,
    ):
        """Step 9: Add a new TTL experience node and connect it to retrieved experiences."""
        self.experiences[exp_id] = exp
        self.node_qualities[exp_id] = float(exp.get("quality", 0.5))
        self.node_visit_counts[exp_id] = 0
        self._pending_ttl_exps.append(exp)

        new_neighbors: List[Dict] = []
        for pair in scored_pairs:
            nid = pair["neighbor_id"]
            W = pair["W"]
            if W < edge_threshold:
                continue

            edge_k = self.edge_key(exp_id, nid)
            self.initial_weights[edge_k] = W
            self.edge_corrections[edge_k] = 0.0
            self.edge_visit_counts[edge_k] = 0

            edge_info = {
                "neighbor": nid,
                "W": W,
                "S_ent":   pair.get("S_ent", 0.0),
                "S_graph": pair.get("S_graph", 0.0),
                "S_sem":   pair.get("S_sem", 0.0),
                "S_task":  pair.get("S_task", 0.0),
            }
            new_neighbors.append(edge_info)

            reverse_edge = {**edge_info, "neighbor": exp_id}
            self.adjacency.setdefault(nid, []).append(reverse_edge)

        self.adjacency[exp_id] = new_neighbors
        self._adj_dirty = True

    # ------------------------------------------------------------------ Step 10

    def update_weights(
        self,
        delta_t: float,
        retrieved_ids: List[str],
        eta_q0: float = 0.1,
        eta_w0: float = 0.1,
        rho: float = 0.6,
        edge_threshold: float = 0.35,
    ):
        """Step 10: Update node quality and edge correction terms.

        Also materialises a new graph edge when W_effective crosses edge_threshold
        for a pair that previously had no connection (W^0 = 0).

        Args:
            delta_t: Feedback signal (2*score - 1); positive = success.
            retrieved_ids: Ordered list of retrieved experience IDs (rank 0 = best).
            eta_q0: Initial node learning rate.
            eta_w0: Initial edge learning rate.
            rho: Decay exponent in (0.5, 1].
            edge_threshold: Minimum effective W to materialise a new edge.
        """
        K = [eid for eid in retrieved_ids if eid in self.experiences]
        if not K:
            return

        # ---- credit assignment ----
        s_tilde = {eid: 1.0 / math.log(2 + rank) for rank, eid in enumerate(K)}
        total_s = sum(s_tilde.values())
        a = {eid: s_tilde[eid] / total_s for eid in K}

        pairs = [(K[i], K[j]) for i in range(len(K)) for j in range(i + 1, len(K))]
        b_tilde = {(i, j): a[i] * a[j] for i, j in pairs}
        total_b = sum(b_tilde.values()) or 1.0
        b = {pair: v / total_b for pair, v in b_tilde.items()}

        # ---- node quality update ----
        for eid in K:
            n_i = self.node_visit_counts.get(eid, 0)
            eta_q = eta_q0 / (1 + n_i) ** rho
            q_i = self.node_qualities.get(eid, 0.5)
            self.node_qualities[eid] = max(0.0, min(1.0, q_i + eta_q * a[eid] * delta_t))
            self.node_visit_counts[eid] = n_i + 1

        # ---- edge correction update + materialisation check ----
        for (id1, id2), b_val in b.items():
            key = self.edge_key(id1, id2)
            W0 = self.initial_weights.get(key, 0.0)
            n_ij = self.edge_visit_counts.get(key, 0)
            eta_w = eta_w0 / (1 + n_ij) ** rho
            theta = self.edge_corrections.get(key, 0.0)
            new_theta = theta + eta_w * b_val * delta_t
            self.edge_corrections[key] = new_theta
            self.edge_visit_counts[key] = n_ij + 1

            # Materialise a new edge when effective W crosses threshold
            W_new = max(0.0, min(1.0, W0 + new_theta))
            if W0 < edge_threshold <= W_new:
                self._materialize_edge(id1, id2, W_new)

    # ------------------------------------------------------------------ helpers

    def _materialize_edge(self, id1: str, id2: str, W: float) -> None:
        """Create a new edge in the working copies when θ pushes W over threshold.

        Sets the effective W as the new baseline (W^0 = W, θ = 0) so future
        corrections accumulate cleanly from this point.
        """
        from src.shared.logger import logger
        key = self.edge_key(id1, id2)

        # Reset: treat current effective W as the new initial weight
        self.initial_weights[key] = W
        self.edge_corrections[key] = 0.0

        edge_info = {
            "neighbor": id2, "W": W,
            "S_ent": 0.0, "S_graph": 0.0, "S_sem": 0.0, "S_task": 0.0,
        }
        self.adjacency.setdefault(id1, []).append(edge_info)
        self.adjacency.setdefault(id2, []).append({**edge_info, "neighbor": id1})
        self._adj_dirty = True

        self._new_materialized_edges.append({
            "src": id1, "dst": id2,
            "W": W, "S_ent": 0.0, "S_graph": 0.0, "S_sem": 0.0, "S_task": 0.0,
            "short_reason": "ttl_materialized",
        })

        logger.log_info(
            f"  [TTL Step 10] New edge materialised: {id1} ↔ {id2}  W={W:.4f}"
        )
