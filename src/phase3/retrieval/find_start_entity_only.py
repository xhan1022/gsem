"""Entity-only starting-node finder for ablation experiments."""
from typing import Any, Dict, List

from find_start_two_stage import FindStartTwoStage, log


class FindStartEntityOnly(FindStartTwoStage):
    """Ablation: remove Stage 2 semantic recall, keep entity recall only."""

    def find(self, case: Dict[str, Any]) -> List[str]:
        active = self._active_experiences()
        if not active:
            log.warning("No active experiences — returning empty list")
            return []

        query_entities = self._extract_query_entities(case)
        hit_count: Dict[str, int] = {}
        hit_map: Dict[str, List[str]] = {}

        active_keys = set(active.keys())
        for ent in query_entities:
            matched = self._entity_to_exps.get(ent, set()) & active_keys
            if matched:
                hit_map[ent] = sorted(matched)
                for eid in matched:
                    hit_count[eid] = hit_count.get(eid, 0) + 1

        if not hit_count:
            log.info("Entity-only retrieval found no matches")
            return []

        for ent, eids in hit_map.items():
            log.info("Entity-only entity='%s' → %d exp(s): %s", ent, len(eids),
                     eids[:5] if len(eids) > 5 else eids)

        ranked = sorted(
            hit_count.keys(),
            key=lambda eid: (
                hit_count[eid],
                float(active[eid].get("quality", 0.0)),
            ),
            reverse=True,
        )
        result = ranked[: self.top_k]
        log.info("Entity-only top-%d start ids: %s", self.top_k, result)
        return result
