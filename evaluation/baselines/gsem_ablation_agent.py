"""Shared ablation baselines for GSEM retrieval experiments."""
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # GSEM/
_RETRIEVAL_DIR = _PROJECT_ROOT / "src" / "phase3" / "retrieval"

if str(_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_RETRIEVAL_DIR))

from find_start_entity_only import FindStartEntityOnly          # noqa: E402
from find_start_semantic_only import FindStartSemanticOnly      # noqa: E402
from find_start_two_stage import FindStartTwoStage              # noqa: E402
from retriver import retrieve as _single_path                   # noqa: E402
from retriver_multi_start import retrieve as _multi_start       # noqa: E402
from retriver_multi_start_no_fill import retrieve as _multi_start_no_fill  # noqa: E402


class _BaseAblationAgent:
    finder_cls = FindStartTwoStage
    retriever = staticmethod(_multi_start)

    def __init__(self, top_k: int = 5, retrieval_model_cfg: Optional[Dict] = None, **kwargs):
        finder_kwargs = {"top_k": top_k}
        if retrieval_model_cfg:
            finder_kwargs.update({
                "deepseek_api_key": retrieval_model_cfg.get("api_key", ""),
                "deepseek_base_url": retrieval_model_cfg.get("base_url", ""),
                "deepseek_model": retrieval_model_cfg.get("model_id", ""),
            })
        self.finder = self.finder_cls(**finder_kwargs)
        self.last_query_entities: list[str] = []

    def retrieve(self, query: str, task_type: Optional[str] = None) -> List[Dict]:
        case = {
            "description": query,
            "task_type": task_type or "diagnosis",
        }
        try:
            experiences = self.retriever(case, self.finder)
            self.last_query_entities = getattr(self.finder, "last_query_entities", [])
            return experiences
        except Exception as e:
            print(f"[{self.__class__.__name__}] Retrieval error (fallback to vanilla): {e}")
            self.last_query_entities = []
            return []


class GSEMSemanticOnlyAgent(_BaseAblationAgent):
    """Ablation 1: semantic-only start finding + multi-start traversal."""

    finder_cls = FindStartSemanticOnly
    retriever = staticmethod(_multi_start)


class GSEMEntityOnlyAgent(_BaseAblationAgent):
    """Ablation 2: entity-only start finding + multi-start traversal."""

    finder_cls = FindStartEntityOnly
    retriever = staticmethod(_multi_start)


class GSEMSinglePathAgent(_BaseAblationAgent):
    """Ablation 3: two-stage start finding + single-path traversal."""

    finder_cls = FindStartTwoStage
    retriever = staticmethod(_single_path)


class GSEMMultiStartNoFillAgent(_BaseAblationAgent):
    """Ablation 4: multi-start traversal without per-start fallback-to-start."""

    finder_cls = FindStartTwoStage
    retriever = staticmethod(_multi_start_no_fill)
