"""TTL: Online Evolution Pipeline — 10 steps per case.

Steps 1-3 are new (this module + reasoning_agent + experience_extractor).
Steps 4-10 directly reuse Phase 1 ERV and Phase 2 graph-construction modules.

Step mapping:
  1  ReasoningAgent.run()                  → trajectory + retrieved_pairs
  2  Stage1Rollout evaluator               → is_correct
  3  OnlineExperienceExtractor.extract()   → new_exp
  4  Stage6ERV (replay sampling)           → Q_0
  5  CoreEntityExtractor.process_single()  → core_entities
  6  normalize_form() + lexicon lookup     → canonical_entities
  7  RoleEdgeExtractor.process_single()    → role_edges, entity_edges
  8  inline TF-IDF / Jaccard / LLM / task  → scored_pairs
  9  GraphState.add_experience()           → updated graph
  10 GraphState.update_weights()           → updated quality + edge corrections
"""
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.shared.config import config
from src.shared.logger import logger
from src.phase1.prompts import EVALUATION_SYSTEM_PROMPT, EVALUATION_HUMAN_PROMPT
from src.phase1.stages.erv import Stage6ERV
from src.phase2.graph.entity_extraction import CoreEntityExtractor
from src.phase2.graph.structure_extraction import RoleEdgeExtractor
from src.phase2.graph.entity_normalizer import normalize_form
from src.phase2.graph.prompts import (
    SEMANTIC_SIMILARITY_SYSTEM_PROMPT,
    SEMANTIC_SIMILARITY_HUMAN_PROMPT,
)

from .retrieval_tool import BaseRetrievalInterface, StubRetrievalInterface
from .reasoning_agent import ReasoningAgent
from .experience_extractor import OnlineExperienceExtractor
from .graph_state import GraphState, TTL_DIR_DEFAULT


class OnlineEvolutionPipeline:
    """Orchestrates TTL online evolution — 10 steps per case."""

    def __init__(
        self,
        retrieval_interface: BaseRetrievalInterface = None,
        graph_state: GraphState = None,
        top_k: int = 5,
        erv_samples: int = 5,
        edge_threshold: float = 0.35,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.40,
        delta: float = 0.10,
        eta_q0: float = 0.10,
        eta_w0: float = 0.10,
        rho: float = 0.60,
        entity_stats_path: str = "data/entity_stats/entity_stats.json",
        lexicon_path: str = "data/lexicon/lexicon.json",
        entity_postings_path: str = "data/entity_postings/entity_postings.json",
        ttl_dir: str = TTL_DIR_DEFAULT,
    ):
        retrieval = retrieval_interface or StubRetrievalInterface()
        self.graph_state = graph_state or GraphState.load_from_static()
        self.edge_threshold = edge_threshold
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.eta_q0, self.eta_w0, self.rho = eta_q0, eta_w0, rho
        self.ttl_dir = ttl_dir

        # ---- Step 1: Reasoning agent ----
        self.reasoning_agent = ReasoningAgent(
            retrieval_interface=retrieval, top_k=top_k
        )

        # ---- Step 2: Evaluator (reuse Phase 1 evaluation prompt) ----
        self.eval_llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATION_SYSTEM_PROMPT),
            ("human", EVALUATION_HUMAN_PROMPT),
        ])

        # ---- Step 3: Experience extractor ----
        self.exp_extractor = OnlineExperienceExtractor()

        # ---- Step 4: ERV (reuse Stage6ERV, override sampling count) ----
        self.erv = Stage6ERV()
        self.erv.sampling_count = erv_samples   # TTL uses 5 samples (tech doc)

        # ---- Steps 5 & 7: Phase 2 extractors ----
        self.entity_extractor = CoreEntityExtractor()
        self.structure_extractor = RoleEdgeExtractor()

        # ---- Step 6: Lexicon for entity normalisation ----
        # Read-only bootstrap: prefer TTL lexicon if it exists, else fallback to static.
        self.ttl_lexicon_path = os.path.join(ttl_dir, "lexicon", "lexicon.json")
        self.lexicon: Dict[str, str] = {}
        self._lexicon_dirty = False
        _lex_source = self.ttl_lexicon_path if os.path.exists(self.ttl_lexicon_path) else lexicon_path
        if os.path.exists(_lex_source):
            with open(_lex_source, encoding="utf-8") as f:
                lex = json.load(f)
                self.lexicon = lex.get("alias_to_canonical", {})
            logger.log_info(f"  [TTL] Loaded lexicon ({_lex_source}): {len(self.lexicon)} aliases")

        # ---- Entity postings for TTL entity-graph export ----
        self.ttl_entity_postings_path = os.path.join(
            ttl_dir, "entity_postings", "entity_postings.json"
        )
        self.entity_postings: Dict[str, List[str]] = {}
        self._entity_postings_dirty = False
        _postings_source = (
            self.ttl_entity_postings_path
            if os.path.exists(self.ttl_entity_postings_path)
            else entity_postings_path
        )
        if os.path.exists(_postings_source):
            with open(_postings_source, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.entity_postings = {
                        str(k): [str(x) for x in v] if isinstance(v, list) else []
                        for k, v in loaded.items()
                    }
            logger.log_info(
                f"  [TTL] Loaded entity postings ({_postings_source}): "
                f"{len(self.entity_postings)} entities"
            )

        # ---- Step 8: Entity IDF for S_ent ----
        self.entity_idf: Dict[str, float] = {}
        self.entity_N: int = 1
        if os.path.exists(entity_stats_path):
            with open(entity_stats_path, encoding="utf-8") as f:
                stats = json.load(f)
                self.entity_idf = stats.get("idf", {})
                self.entity_N = stats.get("N", 1)
            logger.log_info(f"  [TTL] Loaded entity IDF: {len(self.entity_idf)} entries")

        # ---- Step 8: Semantic similarity LLM (reuses Phase 2 prompt) ----
        self.sem_llm = ChatOpenAI(
            model_name=config.deepseek.model_name,
            openai_api_base=config.deepseek.base_url,
            openai_api_key=config.deepseek.api_key,
            temperature=0.0,
        )
        self.sem_prompt = ChatPromptTemplate.from_messages([
            ("system", SEMANTIC_SIMILARITY_SYSTEM_PROMPT),
            ("human", SEMANTIC_SIMILARITY_HUMAN_PROMPT),
        ])

    # ==================================================================
    # Main entry
    # ==================================================================

    def process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run 10-step TTL online evolution for a single case."""
        case_id = case.get("case_id", "?")
        logger.log_info(f"\n{'='*60}")
        logger.log_info(f"[TTL] Case: {case_id}")
        logger.log_info(f"{'='*60}")

        # ----------------------------------------------------------
        # Step 1: Inference with retrieved experiences
        # ----------------------------------------------------------
        trajectory, retrieved_pairs = self.reasoning_agent.run(case, self.graph_state)
        retrieved_ids = [eid for eid, _ in retrieved_pairs]

        # ----------------------------------------------------------
        # Step 2: Evaluate correctness
        # ----------------------------------------------------------
        is_correct = self._evaluate(trajectory, case.get("answer", ""))
        score = 1 if is_correct else 0
        logger.log_info(f"  [Step 2] Correct: {is_correct}")

        # ----------------------------------------------------------
        # Step 3: Extract one experience from trajectory
        # ----------------------------------------------------------
        new_exp = self.exp_extractor.extract(case, trajectory, is_correct)
        if new_exp is None:
            logger.log_warning("[TTL Step 3] No experience extracted — skipping Steps 4-9")
            if retrieved_ids:
                self.graph_state.update_weights(
                    2 * score - 1, retrieved_ids, self.eta_q0, self.eta_w0, self.rho,
                    edge_threshold=self.edge_threshold,
                )
            return {
                "case_id": case_id,
                "score": score,
                "new_experience_id": None,
                "retrieved_ids": retrieved_ids,
            }

        logger.log_info(f"  [Step 3] Extracted {new_exp['type']} experience")

        # ----------------------------------------------------------
        # Step 4: ERV — compute Q_0 for the new TTL experience
        # ----------------------------------------------------------
        a_erv, Q_0 = self._run_erv(case, new_exp)
        new_exp["quality"] = Q_0
        logger.log_info(f"  [Step 4] ERV: a_erv={a_erv:.2f}, Q_0={Q_0:.4f}")

        # Assign TTL experience ID
        new_id = self.graph_state.next_exp_id()
        new_exp["id"] = new_id
        new_exp["task_type_norm"] = new_exp.get("task_type", "").strip().lower()

        # ----------------------------------------------------------
        # Step 5: Core entity extraction (reuse Phase 2)
        # ----------------------------------------------------------
        new_exp = self.entity_extractor.process_single(new_exp)
        logger.log_info(f"  [Step 5] Entities: {len(new_exp.get('core_entities', []))}")
        self._append_intermediate(
            "core_entities/core_entities.jsonl",
            {k: new_exp[k] for k in new_exp if k not in ("entity_edges",)},
        )

        # ----------------------------------------------------------
        # Step 6: Entity normalisation — form norm + lexicon lookup
        # ----------------------------------------------------------
        new_exp = self._normalise_entities(new_exp)
        logger.log_info(
            f"  [Step 6] Canonical entities: {len(new_exp.get('canonical_entities', []))}"
        )
        self._update_entity_postings(new_id, new_exp.get("canonical_entities", []))

        # ----------------------------------------------------------
        # Step 7: Role-edge structure extraction (reuse Phase 2)
        # ----------------------------------------------------------
        new_exp = self.structure_extractor.process_single(new_exp)
        logger.log_info(f"  [Step 7] Role-edges: {new_exp.get('role_edges', [])}")
        self._append_intermediate("role_edges/role_edges.jsonl", new_exp)

        # ----------------------------------------------------------
        # Step 8: Similarity vs retrieved set only (4 dimensions)
        # 无召回时跳过，新经验作为孤立节点加入图，Step 10 同样跳过
        # ----------------------------------------------------------
        if not retrieved_pairs:
            scored_pairs = []
            logger.log_info("  [Step 8] 跳过（无召回经验），新经验将作为孤立节点加入图")
        else:
            scored_pairs = self._compute_similarity(new_exp, retrieved_pairs)
            logger.log_info(f"  [Step 8] Scored pairs: {len(scored_pairs)}")
            self._append_intermediate(
                "similarity/similarity.jsonl",
                {
                    "new_exp_id": new_id,
                    "case_id": case_id,
                    "retrieved_count": len(retrieved_pairs),
                    "scored_pairs": scored_pairs,
                },
            )

        # ----------------------------------------------------------
        # Step 9: Merge new TTL experience into graph
        # ----------------------------------------------------------
        self.graph_state.add_experience(
            new_id, new_exp, scored_pairs, self.edge_threshold
        )
        logger.log_info(f"  [Step 9] Merged {new_id} into graph")
        # Save adjacency for this new node (Phase 2 compatible format)
        self._append_intermediate(
            "experience_graph_adj/experience_graph_adj.jsonl",
            {
                "exp_id": new_id,
                "neighbors": self.graph_state.adjacency.get(new_id, []),
            },
        )

        # ----------------------------------------------------------
        # Step 10: Update retrieved-experience weights
        # ----------------------------------------------------------
        if retrieved_ids:
            delta_t = 2 * score - 1
            self.graph_state.update_weights(
                delta_t, retrieved_ids, self.eta_q0, self.eta_w0, self.rho,
                edge_threshold=self.edge_threshold,
            )
            logger.log_info(
                f"  [Step 10] Updated weights: delta_t={delta_t:+.1f}, "
                f"{len(retrieved_ids)} nodes"
            )

        # Persist lexicon if new entries were added this case
        self._save_lexicon()
        self._save_entity_postings()

        return {
            "case_id": case_id,
            "score": score,
            "new_experience_id": new_id,
            "a_erv": a_erv,
            "Q_0": Q_0,
            "retrieved_ids": retrieved_ids,
        }

    # ==================================================================
    # Step 2 helpers
    # ==================================================================

    def _evaluate(self, trajectory: Dict[str, Any], gold_answer: str) -> bool:
        """Reuse Phase 1 evaluation prompt to check correctness."""
        final_answer = trajectory.get("final_answer", "")
        if not final_answer.strip():
            return False
        try:
            prompt = self.eval_prompt.format_messages(
                total_steps=0,
                final_answer=final_answer,
                gold_standard=gold_answer,
            )
            response = self.eval_llm.invoke(prompt)
            content = response.content.strip()
            first = content.find("{")
            last = content.rfind("}")
            if first == -1 or last == -1:
                return False
            evaluation = json.loads(content[first: last + 1])
            return bool(evaluation.get("success", False))
        except Exception as e:
            logger.log_warning(f"[TTL Step 2] Evaluation error: {e}")
            return False

    # ==================================================================
    # Step 4 helpers
    # ==================================================================

    def _run_erv(
        self, case: Dict[str, Any], new_exp: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Run ERV for a single TTL experience.

        Returns (a_erv, Q_0).
        """
        a_erv = self.erv.replay_sampling(case, [new_exp])
        delta_r = a_erv - self.erv.threshold
        Q_0 = self.erv.map_delta_to_quality(delta_r)
        return a_erv, Q_0

    # ==================================================================
    # Step 6 helpers
    # ==================================================================

    def _normalise_entities(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Form normalisation + lexicon lookup for core entities.

        Entities found in the TTL lexicon resolve to their canonical form.
        Unknown entities are added to the lexicon as self-canonicals (form → form)
        so subsequent cases can align to the same term.
        """
        canonical_set = set()
        new_entries: Dict[str, str] = {}

        for item in exp.get("core_entities", []):
            raw = item.get("entity", "") if isinstance(item, dict) else str(item)
            if not raw:
                continue
            form = normalize_form(raw)
            if raw in self.lexicon:
                canonical = self.lexicon[raw]
            elif form in self.lexicon:
                canonical = self.lexicon[form]
            else:
                # New entity — add raw and form as self-canonical
                canonical = form
                new_entries[raw] = form
                new_entries[form] = form
            canonical_set.add(canonical)

        if new_entries:
            self.lexicon.update(new_entries)
            self._lexicon_dirty = True
            logger.log_info(
                f"  [Step 6] Lexicon +{len(new_entries)} new aliases "
                f"(total: {len(self.lexicon)})"
            )

        exp["canonical_entities"] = sorted(canonical_set)
        return exp

    # ==================================================================
    # Step 8 helpers — similarity vs retrieved set
    # ==================================================================

    def _compute_similarity(
        self,
        new_exp: Dict[str, Any],
        retrieved_pairs: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Compute 4-dim similarity between new_exp and each retrieved experience."""
        results = []
        for eid, ret_exp in retrieved_pairs:
            s_ent = self._s_ent(new_exp, ret_exp)
            s_graph = self._s_graph(new_exp, ret_exp)
            s_sem = self._s_sem(new_exp, ret_exp)
            s_task = self._s_task(new_exp, ret_exp)
            W = (
                self.alpha * s_ent
                + self.beta * s_graph
                + self.gamma * s_sem
                + self.delta * s_task
            )
            results.append({
                "neighbor_id": eid,
                "S_ent": round(s_ent, 4),
                "S_graph": round(s_graph, 4),
                "S_sem": round(s_sem, 4),
                "S_task": round(s_task, 4),
                "W": round(W, 4),
            })
        return results

    def _tfidf_vector(self, entities: List[str]) -> Dict[str, float]:
        tf: Dict[str, int] = defaultdict(int)
        for e in entities:
            tf[e] += 1
        total = len(entities) or 1
        vec: Dict[str, float] = {}
        for e, cnt in tf.items():
            idf = self.entity_idf.get(e, math.log(self.entity_N + 1))
            vec[e] = (cnt / total) * idf
        return vec

    @staticmethod
    def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        shared = set(v1) & set(v2)
        if not shared:
            return 0.0
        dot = sum(v1[e] * v2[e] for e in shared)
        norm1 = math.sqrt(sum(x * x for x in v1.values()))
        norm2 = math.sqrt(sum(x * x for x in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _s_ent(self, exp_a: Dict, exp_b: Dict) -> float:
        v1 = self._tfidf_vector(exp_a.get("canonical_entities", []))
        v2 = self._tfidf_vector(exp_b.get("canonical_entities", []))
        return self._cosine(v1, v2)

    @staticmethod
    def _s_graph(exp_a: Dict, exp_b: Dict) -> float:
        set_a = set(exp_a.get("role_edges", []))
        set_b = set(exp_b.get("role_edges", []))
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 0.0

    def _s_sem(self, exp_a: Dict, exp_b: Dict) -> float:
        """LLM semantic similarity — reuses Phase 2 SEMANTIC_SIMILARITY prompts."""
        try:
            prompt = self.sem_prompt.format_messages(
                condition_a=exp_a.get("condition", ""),
                content_a=exp_a.get("content", ""),
                condition_b=exp_b.get("condition", ""),
                content_b=exp_b.get("content", ""),
            )
            response = self.sem_llm.invoke(prompt)
            content = response.content.strip()
            m = re.search(r'"similarity"\s*:\s*([0-9.]+)', content)
            if m:
                return max(0.0, min(1.0, float(m.group(1))))
            m = re.search(r'\b(0\.\d+|1\.0+|0|1)\b', content)
            if m:
                return max(0.0, min(1.0, float(m.group(1))))
        except Exception as e:
            logger.log_warning(f"[TTL Step 8] S_sem LLM call failed: {e}")
        return 0.0

    @staticmethod
    def _s_task(exp_a: Dict, exp_b: Dict) -> float:
        t_a = exp_a.get("task_type_norm", exp_a.get("task_type", "")).strip().lower()
        t_b = exp_b.get("task_type_norm", exp_b.get("task_type", "")).strip().lower()
        return 1.0 if t_a and t_b and t_a == t_b else 0.0

    # ==================================================================
    # Intermediate data persistence
    # ==================================================================

    def _append_intermediate(self, rel_path: str, record: Dict) -> None:
        """Append a JSON record to data/ttl/<rel_path>, creating dirs as needed."""
        path = os.path.join(self.ttl_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_lexicon(self) -> None:
        """Persist TTL lexicon to data/ttl/lexicon/lexicon.json if updated."""
        if not self._lexicon_dirty:
            return
        os.makedirs(os.path.dirname(self.ttl_lexicon_path), exist_ok=True)
        # Rebuild canonical_to_aliases from current alias_to_canonical
        from collections import defaultdict as _dd
        c2a: Dict[str, list] = _dd(list)
        for alias, canonical in self.lexicon.items():
            if alias not in c2a[canonical]:
                c2a[canonical].append(alias)
        lexicon_out = {
            "alias_to_canonical": self.lexicon,
            "canonical_to_aliases": {k: sorted(v) for k, v in c2a.items()},
        }
        with open(self.ttl_lexicon_path, "w", encoding="utf-8") as f:
            json.dump(lexicon_out, f, ensure_ascii=False, indent=2)
        self._lexicon_dirty = False
        logger.log_info(
            f"  [TTL] Lexicon saved → {self.ttl_lexicon_path} "
            f"({len(self.lexicon)} aliases)"
        )

    def _update_entity_postings(self, exp_id: str, canonical_entities: List[str]) -> None:
        """Incrementally maintain entity -> exp_ids postings for TTL entity graph export."""
        changed = 0
        for ent in canonical_entities or []:
            if not ent:
                continue
            posting = self.entity_postings.setdefault(ent, [])
            if exp_id not in posting:
                posting.append(exp_id)
                changed += 1
        if changed:
            self._entity_postings_dirty = True
            logger.log_info(
                f"  [Step 6] Entity postings +{changed} links "
                f"(entities: {len(self.entity_postings)})"
            )

    def _save_entity_postings(self) -> None:
        """Persist TTL entity postings to data/ttl/entity_postings/entity_postings.json."""
        if not self._entity_postings_dirty:
            return
        os.makedirs(os.path.dirname(self.ttl_entity_postings_path), exist_ok=True)
        with open(self.ttl_entity_postings_path, "w", encoding="utf-8") as f:
            json.dump(self.entity_postings, f, ensure_ascii=False, indent=2)
        self._entity_postings_dirty = False
        logger.log_info(
            f"  [TTL] Entity postings saved → {self.ttl_entity_postings_path} "
            f"({len(self.entity_postings)} entities)"
        )
