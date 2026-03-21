"""Semantic-only starting-node finder for ablation experiments."""
from typing import Any, Dict, List

from find_start_two_stage import FindStartTwoStage, log


class FindStartSemanticOnly(FindStartTwoStage):
    """Ablation: remove Stage 1 entity recall, keep semantic top-k only."""

    def find(self, case: Dict[str, Any]) -> List[str]:
        self._ensure_embeddings_built()
        active = self._active_experiences()
        if not active:
            log.warning("No active experiences — returning empty list")
            return []

        active_ids = list(active.keys())
        query_text = case.get("description", "")
        query_vec = self._embed_one(query_text)

        sem_score = {
            eid: self._cosine_lists(query_vec, self._embeddings.get(eid, []))
            for eid in active_ids
        }
        result = sorted(active_ids, key=lambda e: sem_score[e], reverse=True)[: self.top_k]

        log.info("Semantic-only top-%d start ids: %s", self.top_k, result)
        return result
