"""GSEM baseline — graph traversal retrieval (FindStartTwoStage + retriver.py).

retrieve() 返回空列表时，ReActAgent 自动退化为用自身模型能力推理（vanilla ReAct），
不需要在此处额外处理。
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # GSEM/
_RETRIEVAL_DIR = _PROJECT_ROOT / "src" / "phase3" / "retrieval"

# retriver.py 内部有 `from agent import call_llm`，需要把 src/phase3/retrieval/ 加入 sys.path
if str(_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_RETRIEVAL_DIR))

from find_start_two_stage import FindStartTwoStage       # noqa: E402
from retriver_multi_start import retrieve as _graph_retrieve  # noqa: E402


class GSEMAgent:
    """
    GSEM 检索 baseline。

    retrieve(query, task_type) → List[Dict]
      - 非空：经验注入 ReAct prompt
      - 空列表：ReActAgent 退化为 vanilla ReAct，用自身模型能力推理
    """

    def __init__(self, top_k: int = 5, retrieval_model_cfg: Optional[Dict] = None, **kwargs):
        """
        Args:
            top_k:   FindStartTwoStage 返回的起始节点数量。
            **kwargs: 兼容旧接口的多余参数（experience_file、graph_data_dir），忽略。
        """
        finder_kwargs = {"top_k": top_k}
        if retrieval_model_cfg:
            finder_kwargs.update({
                "deepseek_api_key": retrieval_model_cfg.get("api_key", ""),
                "deepseek_base_url": retrieval_model_cfg.get("base_url", ""),
                "deepseek_model": retrieval_model_cfg.get("model_id", ""),
            })
        self.finder = FindStartTwoStage(**finder_kwargs)
        self.last_query_entities: list[str] = []

    def retrieve(self, query: str, task_type: Optional[str] = None) -> List[Dict]:
        """
        对 query 做图遍历检索，返回 agent 选中的经验列表。
        失败或无召回时返回 []，上层 ReActAgent 会用 vanilla ReAct 兜底。
        """
        case = {
            "description": query,
            "task_type": task_type or "diagnosis",
        }
        try:
            experiences = _graph_retrieve(case, self.finder)
            self.last_query_entities = getattr(self.finder, "last_query_entities", [])
            return experiences
        except Exception as e:
            print(f"[GSEMAgent] Retrieval error (fallback to vanilla): {e}")
            self.last_query_entities = []
            return []
