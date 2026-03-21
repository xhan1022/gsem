"""Retrieval Step 1 (Two-Stage): FindStartTwoStage.

Stage 1  Sparse retrieval via entity graph
         - Extract + normalise query entities
         - Look each entity up in the entity graph (exact canonical-form match)
         - Union all exp_ids found → candidate pool
         - If no candidates found, fall back to full experience set

Stage 2  Dense retrieval within candidate pool
         - Compute embedding cosine similarity between query and each candidate
         - Return top-k as final starting nodes

Public interface:

    finder = FindStartTwoStage(static_only=True)
    start_ids = finder.find(case)

Data files (in src/phase3/retrieval/data/):
  entity_graph.json   entity graph (copy via FindStartTwoStage.setup_data())
  experiences.jsonl   experience library
  lexicon.json        normalisation lexicon
  embeddings.jsonl    embedding cache
"""
import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Inline BM25 (no external dependency)
# ===========================================================================

def _tokenize(text: str) -> List[str]:
    """Tokenize text — handles both English words and CJK characters."""
    text = text.lower()
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)


class _SimpleBM25:
    """Minimal BM25 implementation (Robertson & Zaragoza, 2009)."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.N if self.N else 1.0

        # df: term → number of documents containing it
        df: Dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self.idf: Dict[str, float] = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        self._corpus = corpus

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.N
        for term in query:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self._corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = len(doc)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Module-local logger
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        format="%(asctime)s [FindStartTwoStage] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

# ---------------------------------------------------------------------------
# Self-contained data directory
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_MODULE_DIR, "data")

DEFAULT_ENTITY_GRAPH_PATH = os.path.join(_DATA_DIR, "entity_graph.json")
DEFAULT_EXPERIENCES_PATH  = os.path.join(_DATA_DIR, "experiences.jsonl")
DEFAULT_LEXICON_PATH      = os.path.join(_DATA_DIR, "lexicon.json")
DEFAULT_EMBEDDING_CACHE   = os.path.join(_DATA_DIR, "embeddings.jsonl")

_FALLBACK_ENTITY_GRAPH_PATHS = [
    os.path.join("src", "retrieval", "graph", "entity_graph.json"),
]
_FALLBACK_LEXICON_PATHS = [
    os.path.join("data", "ttl", "lexicon", "lexicon.json"),
    os.path.join("data", "lexicon", "lexicon.json"),
]
_FALLBACK_EXPERIENCES_PATHS = [
    os.path.join(_DATA_DIR, "experience.jsonl"),          # singular (actual file)
    os.path.join("data", "ttl", "experiences", "experiences.jsonl"),
    os.path.join("data", "experiences", "experiences.jsonl"),
]

_EMBED_BATCH_SIZE = 10


# ===========================================================================
# Inline: form normalisation (copied from src/graph/entity_normalizer.py)
# ===========================================================================

def _normalize_form(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:!?()[]{}'\" ")
    text = re.sub(r"\s*-\s*", "-", text)
    return text


# ===========================================================================
# Inline: entity extraction prompts (copied from src/graph/prompts.py)
# ===========================================================================

_ENTITY_EXTRACTION_SYSTEM = """\
You are extracting a compact set of medical decision-structure entities from a reusable clinical reasoning experience.

Goal:
Extract only the most decision-driving medical entities that define the clinical situation and the actionable strategy, while keeping the entity list small and easy to normalize.

Role schema (fixed):
- Condition: diagnoses, symptoms, findings, clinical states, patient status.
- Constraint: contraindications, feasibility limits, special risks, "cannot / not suitable" factors.
- Action: diagnostic tests, treatments, interventions, procedures, management steps.
- Rationale: medical reasoning basis, mechanism, justification, key rationale statements.
- Outcome: intended clinical goal or effect (e.g., prevent complication, control symptoms, enable recovery).

Role schema (fixed) and how to choose:
- Condition: keep 1–2 anchors that determine whether the strategy applies (typically one core diagnosis, key presentation, or key population feature).
- Constraint: keep at most 1 anchor that makes the default approach infeasible or substantially increases risk; if none, omit this role.
- Action: keep 1–2 anchors that truly change the decision (the main action plus one key alternative or critical adjunct if needed).
- Rationale: keep at most 1 anchor that explains why the strategy holds (mechanism/causal link/critical disambiguation); if none, omit this role.
- Outcome: keep at most 1 anchor describing the intent or expected benefit (e.g., control output, avoid misdiagnosis, enable rehabilitation); if none, omit this role.

Extraction rules:
- Extract only medically meaningful, decision-driving anchor entities; do not aim for exhaustive coverage.
- Use standard medical terminology; avoid generic words.
- **CRITICAL: Each entity MUST be a noun or noun phrase of EXACTLY 1-3 words. No exceptions.**
  - Good examples: "myocardial infarction" (2 words), "heart failure" (2 words), "COPD" (1 word)
  - Bad examples: "optimize wound healing environment" (4 words, too long), "prevent aspiration" (verb phrase, not noun)
- Do not duplicate the same meaning across roles.
- Each entity must be assigned to exactly one role from the fixed schema.
- Keep the total set compact (typically 5–8 entities); omit roles not supported by the experience rather than forcing them.

Output format:
Return a JSON object with a single field "core_entities" as a list.
Each item must have "entity" and "role".

Example format:
{{
  "core_entities": [
    {{"entity": "...", "role": "Condition"}},
    {{"entity": "...", "role": "Action"}}
  ]
}}\
"""

_ENTITY_EXTRACTION_HUMAN = """\
## Condition
{condition}

## Content
{content}

Extract the core decision-structure entities and roles as specified in the system prompt.

Return a valid JSON object with field 'core_entities' containing a list of entity-role pairs.
Output only the JSON, no other text.\
"""


# ===========================================================================
# Inline: lightweight entity extractor
# ===========================================================================

class _EntityExtractor:
    def __init__(self, api_key: str, base_url: str, model: str):
        self._llm = ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.0,
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _ENTITY_EXTRACTION_SYSTEM),
            ("human",  _ENTITY_EXTRACTION_HUMAN),
        ])
        self._chain = self._prompt | self._llm | JsonOutputParser()

    def extract(self, condition: str, content: str = "") -> List[Dict]:
        try:
            resp = self._chain.invoke({"condition": condition, "content": content})
            return [
                d for d in resp.get("core_entities", [])
                if isinstance(d, dict) and "entity" in d and "role" in d
            ]
        except Exception as exc:
            log.warning("Entity extraction failed: %s", exc)
            return []


# ===========================================================================
# Main class
# ===========================================================================

class FindStartTwoStage:
    """Two-stage starting-node finder.

    Stage 1 (sparse):  entity graph lookup → candidate pool
    Stage 2 (dense):   embedding cosine similarity within pool → top-k

    Drop-in replacement for FindStart — same find(case) interface.

    Args:
        experiences:          exp_id → exp_dict. If None, loaded from experiences_path.
        experiences_path:     JSONL file to load experiences from.
        top_k:                Final number of starting nodes.
        static_only:          True → ignore TTL experiences (source=="ttl").
        entity_graph_path:    Path to entity_graph.json.
        lexicon_path:         Path to lexicon.json.
        embedding_cache_path: Path to embeddings.jsonl (shared with FindStart).
    """

    def __init__(
        self,
        experiences: Dict[str, Dict] = None,
        experiences_path: str = DEFAULT_EXPERIENCES_PATH,
        top_k: int = 5,
        static_only: bool = False,
        # DeepSeek (entity extraction)
        deepseek_api_key: str = "",
        deepseek_base_url: str = "",
        deepseek_model: str = "",
        # Embedding API
        embedding_api_key: str = "",
        embedding_base_url: str = "",
        embedding_model: str = "",
        # Data paths
        entity_graph_path: str = DEFAULT_ENTITY_GRAPH_PATH,
        lexicon_path: str = DEFAULT_LEXICON_PATH,
        embedding_cache_path: str = DEFAULT_EMBEDDING_CACHE,
    ):
        self.top_k = top_k
        self.static_only = static_only
        self.embedding_cache_path = embedding_cache_path

        # ---- Experiences ----
        if experiences is not None:
            self.experiences = experiences
        else:
            self.experiences = self._load_experiences(experiences_path)

        # ---- API credentials ----
        dk_key  = deepseek_api_key  or os.getenv("DEEPSEEK_API_KEY", "")
        dk_url  = deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        dk_mod  = deepseek_model    or os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
        emb_key = embedding_api_key  or os.getenv("EMBEDDING_API_KEY", "")
        emb_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        emb_mod = embedding_model    or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self._embed_model = emb_mod

        # ---- Read-only lexicon ----
        self.lexicon: Dict[str, str] = {}
        lex_path = self._resolve_path(lexicon_path, _FALLBACK_LEXICON_PATHS, "lexicon")
        if lex_path:
            with open(lex_path, encoding="utf-8") as f:
                self.lexicon = json.load(f).get("alias_to_canonical", {})
            log.info("Lexicon loaded (%s): %d aliases", lex_path, len(self.lexicon))

        # ---- Entity graph: entity_id → set of exp_ids ----
        self._entity_to_exps: Dict[str, Set[str]] = {}
        eg_path = self._resolve_path(entity_graph_path, _FALLBACK_ENTITY_GRAPH_PATHS, "entity_graph")
        if eg_path:
            with open(eg_path, encoding="utf-8") as f:
                eg = json.load(f)
            for node in eg.get("nodes", []):
                eid_norm = _normalize_form(node.get("id", ""))
                if eid_norm:
                    self._entity_to_exps[eid_norm] = set(node.get("exp_ids", []))
            log.info("Entity graph loaded: %d entity nodes", len(self._entity_to_exps))

        # ---- Entity extractor ----
        self._extractor = _EntityExtractor(dk_key, dk_url, dk_mod)

        # ---- Embedding client ----
        self._embed_client = OpenAI(api_key=emb_key, base_url=emb_url)

        # ---- Embedding cache ----
        self._embeddings: Dict[str, List[float]] = {}
        self._ensure_embeddings_built()

        # Last find() metadata (readable by caller after find())
        self.last_query_entities: List[str] = []

    # ==================================================================
    # Public API (same interface as FindStart)
    # ==================================================================

    def find(self, case: Dict[str, Any]) -> List[str]:
        """Return top-k exp_ids as starting nodes.

        Stage 1 (Sparse):  entity graph lookup → candidate pool (may be empty)
        Stage 2 (RAG):     global embedding + BM25 → top-k recall (guaranteed non-empty
                           as long as any active experience exists)
        Merge + Rerank:    union(Stage1, Stage2) → rerank by weighted score → top-k
        """
        self._ensure_embeddings_built()
        active = self._active_experiences()

        if not active:
            log.warning("No active experiences — returning empty list")
            return []

        active_ids = list(active.keys())
        query_text = case.get("description", "")

        # ── Stage 1: sparse entity recall ─────────────────────────────────────
        query_entities = self._extract_query_entities(case)
        stage1_candidates: Set[str] = set()
        hit_map: Dict[str, List[str]] = {}

        for ent in query_entities:
            matched = self._entity_to_exps.get(ent, set()) & set(active.keys())
            if matched:
                hit_map[ent] = sorted(matched)
                stage1_candidates |= matched

        if hit_map:
            for ent, eids in hit_map.items():
                log.info("Stage 1  entity='%s' → %d exp(s): %s", ent, len(eids),
                         eids[:5] if len(eids) > 5 else eids)
        else:
            log.info("Stage 1  no entity matches (stage1_candidates is empty)")

        log.info("Stage 1 candidate pool: %d", len(stage1_candidates))

        # ── Stage 2: global RAG (embedding + BM25) ────────────────────────────
        query_vec = self._embed_one(query_text)

        # 2a. Global embedding scores
        sem_score: Dict[str, float] = {
            eid: self._cosine_lists(query_vec, self._embeddings.get(eid, []))
            for eid in active_ids
        }
        top_sem: Set[str] = set(
            sorted(active_ids, key=lambda e: sem_score[e], reverse=True)[: self.top_k]
        )

        # 2b. Global BM25 scores
        bm25_score: Dict[str, float] = self._bm25_search(query_text, active, active_ids)
        top_bm25: Set[str] = set(
            sorted(active_ids, key=lambda e: bm25_score.get(e, 0.0), reverse=True)[: self.top_k]
        )

        stage2_candidates: Set[str] = top_sem | top_bm25
        log.info("Stage 2 (RAG) emb-top-%d + bm25-top-%d → %d unique",
                 self.top_k, self.top_k, len(stage2_candidates))

        # ── Merge + Rerank ────────────────────────────────────────────────────
        all_candidates: Set[str] = stage1_candidates | stage2_candidates
        log.info("Merged pool: %d candidates (stage1=%d, stage2=%d)",
                 len(all_candidates), len(stage1_candidates), len(stage2_candidates))

        # Normalise BM25 to [0, 1]
        max_bm25 = max(bm25_score.values()) if bm25_score else 1.0
        if max_bm25 == 0.0:
            max_bm25 = 1.0

        reranked: List[Tuple[str, float]] = []
        for eid in all_candidates:
            s_sem   = sem_score.get(eid, 0.0)
            s_bm25  = bm25_score.get(eid, 0.0) / max_bm25
            s_total = 0.50 * s_sem + 0.50 * s_bm25
            reranked.append((eid, s_total))

        reranked.sort(key=lambda x: x[1], reverse=True)
        result = [eid for eid, _ in reranked[: self.top_k]]

        log.info("Reranked top-%d:", self.top_k)
        for eid, score in reranked[: self.top_k]:
            log.info("  [%s] total=%.4f | %s", eid, score,
                     active[eid].get("condition", "")[:60])

        return result

    # ==================================================================
    # Helpers
    # ==================================================================

    def _bm25_search(
        self,
        query_text: str,
        active: Dict[str, Dict],
        active_ids: List[str],
    ) -> Dict[str, float]:
        """Return BM25 score for each active experience against query_text."""
        if not active_ids:
            return {}
        corpus = [_tokenize(self._exp_text(active[eid])) for eid in active_ids]
        bm25 = _SimpleBM25(corpus)
        query_tokens = _tokenize(query_text)
        raw_scores = bm25.get_scores(query_tokens)
        return dict(zip(active_ids, raw_scores))

    def _active_experiences(self) -> Dict[str, Dict]:
        if self.static_only:
            return {eid: exp for eid, exp in self.experiences.items()
                    if exp.get("source") != "ttl"}
        return self.experiences

    def _extract_query_entities(self, case: Dict[str, Any]) -> List[str]:
        description = case.get("description", "")
        raw_list = self._extractor.extract(condition=description)
        canonical: List[str] = []
        for d in raw_list:
            raw = d.get("entity", "")
            if not raw:
                continue
            form  = _normalize_form(raw)
            canon = self.lexicon.get(form, form)
            canonical.append(canon)
        self.last_query_entities = canonical
        log.info("Query entities after normalisation (%d): %s", len(canonical), canonical)
        return canonical

    @staticmethod
    def _cosine_lists(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1  = math.sqrt(sum(a * a for a in v1))
        n2  = math.sqrt(sum(b * b for b in v2))
        return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

    def _embed_one(self, text: str) -> List[float]:
        try:
            resp = self._embed_client.embeddings.create(
                model=self._embed_model, input=text)
            return resp.data[0].embedding
        except Exception as exc:
            log.warning("Embedding API error: %s", exc)
            return []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self._embed_client.embeddings.create(
                model=self._embed_model, input=texts)
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        except Exception as exc:
            log.warning("Batch embedding API error: %s", exc)
            return [[] for _ in texts]

    @staticmethod
    def _exp_text(exp: Dict) -> str:
        return (f"Condition: {exp.get('condition', '')}\n"
                f"Content: {exp.get('content', '')}")

    def _ensure_embeddings_built(self) -> None:
        if os.path.exists(self.embedding_cache_path):
            with open(self.embedding_cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        eid = row.get("exp_id", "")
                        emb = row.get("embedding", [])
                        if eid and emb:
                            self._embeddings[eid] = emb

        missing = [eid for eid in self.experiences if eid not in self._embeddings]
        if not missing:
            return

        log.info("Building embeddings for %d experiences (cached: %d) …",
                 len(missing), len(self._embeddings))
        os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)

        with open(self.embedding_cache_path, "a", encoding="utf-8") as cache_f:
            for start in range(0, len(missing), _EMBED_BATCH_SIZE):
                batch_ids   = missing[start : start + _EMBED_BATCH_SIZE]
                batch_texts = [self._exp_text(self.experiences[eid]) for eid in batch_ids]
                embeddings  = self._embed_batch(batch_texts)
                for eid, emb in zip(batch_ids, embeddings):
                    if emb:
                        self._embeddings[eid] = emb
                        cache_f.write(
                            json.dumps({"exp_id": eid, "embedding": emb},
                                       ensure_ascii=False) + "\n")
                done = min(start + _EMBED_BATCH_SIZE, len(missing))
                log.info("  %d / %d embeddings done …", done, len(missing))

        log.info("Embedding cache ready: %d total → %s",
                 len(self._embeddings), self.embedding_cache_path)

    @staticmethod
    def _load_experiences(path: str) -> Dict[str, Dict]:
        exps: Dict[str, Dict] = {}
        candidates = [path] + _FALLBACK_EXPERIENCES_PATHS
        for p in candidates:
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            exp = json.loads(line)
                            eid = exp.get("id", "")
                            if eid:
                                exps[eid] = exp
                log.info("Experiences loaded from %s: %d", p, len(exps))
                return exps
        log.warning("Experiences file not found (checked: %s)", candidates)
        return exps

    @staticmethod
    def _resolve_path(primary: str, fallbacks: List[str], label: str) -> str:
        if os.path.exists(primary):
            return primary
        for fb in fallbacks:
            if os.path.exists(fb):
                log.info("%s not found at %s — using fallback: %s", label, primary, fb)
                return fb
        log.warning("%s not found (checked: %s + fallbacks)", label, primary)
        return ""

    # ==================================================================
    # Setup: copy data files to src/phase3/retrieval/data/
    # ==================================================================

    @classmethod
    def setup_data(cls, entity_graph_src: str = "", lexicon_src: str = "",
                   experiences_src: str = "") -> None:
        """Copy entity_graph, lexicon, experiences to src/phase3/retrieval/data/.

            FindStartTwoStage.setup_data()
        """
        import shutil
        os.makedirs(_DATA_DIR, exist_ok=True)

        pairs = [
            (entity_graph_src, _FALLBACK_ENTITY_GRAPH_PATHS, DEFAULT_ENTITY_GRAPH_PATH, "entity_graph"),
            (lexicon_src,      _FALLBACK_LEXICON_PATHS,       DEFAULT_LEXICON_PATH,      "lexicon"),
            (experiences_src,  _FALLBACK_EXPERIENCES_PATHS,   DEFAULT_EXPERIENCES_PATH,  "experiences"),
        ]
        for src, fallbacks, dst, label in pairs:
            if not src:
                for p in fallbacks:
                    if os.path.exists(p):
                        src = p
                        break
            if src and os.path.exists(src):
                shutil.copy2(src, dst)
                log.info("Copied %s → %s", label, dst)
            else:
                log.warning("%s source not found; skipping.", label)
