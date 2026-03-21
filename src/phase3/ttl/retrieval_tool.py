"""TTL: Experience Retrieval Interface.

Defines the retrieval contract for TTL. Implement BaseRetrievalInterface
to integrate a real retrieval module.  The interface can also be exposed as a
LangChain StructuredTool for ReAct agent integration.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .graph_state import GraphState


class BaseRetrievalInterface(ABC):
    """Abstract retrieval interface. Subclass to integrate a real retrieval module."""

    @abstractmethod
    def retrieve(
        self,
        case: Dict[str, Any],
        graph_state: "GraphState",
        top_k: int = 5,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Retrieve top-k relevant experiences for a case.

        Args:
            case: Clinical case dict (contains 'description', 'task_type', etc.)
            graph_state: Current graph state
            top_k: Number of experiences to retrieve

        Returns:
            List of (exp_id, exp_dict) ordered by relevance (highest first).
        """

    def as_langchain_tool(
        self,
        case: Dict[str, Any],
        graph_state: "GraphState",
        top_k: int = 5,
    ) -> StructuredTool:
        """Expose retrieval as a LangChain StructuredTool for ReAct integration.

        The returned tool can be registered in a LangChain ReAct agent so that
        the agent may invoke experience retrieval mid-reasoning.
        """

        class _Input(BaseModel):
            query: str = Field(
                description="Clinical scenario query to retrieve relevant experiences for"
            )

        retrieve_fn = self.retrieve

        def _run(query: str) -> str:
            results = retrieve_fn(case, graph_state, top_k)
            if not results:
                return "No relevant experiences found in the experience graph."
            lines = []
            for eid, exp in results:
                lines.append(
                    f"[{eid}] [{exp.get('type', '?')}]\n"
                    f"  Condition: {exp.get('condition', '')}\n"
                    f"  Guidance:  {exp.get('content', '')}"
                )
            return "\n\n".join(lines)

        return StructuredTool.from_function(
            func=_run,
            name="retrieve_experiences",
            description=(
                "Retrieve relevant clinical reasoning experiences from the experience graph. "
                "Returns top-k experiences matching the given clinical scenario. "
                "Call this tool once at the beginning to get experience-based guidance."
            ),
            args_schema=_Input,
        )


class StubRetrievalInterface(BaseRetrievalInterface):
    """Stub implementation — returns empty until a real module is integrated."""

    def retrieve(
        self,
        case: Dict[str, Any],
        graph_state: "GraphState",
        top_k: int = 5,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        from src.shared.logger import logger

        logger.log_warning(
            "[TTL Retrieval] StubRetrievalInterface active — returning empty list. "
            "Integrate a real retrieval module by subclassing BaseRetrievalInterface."
        )
        return []


class GSEMRetrievalAdapter(BaseRetrievalInterface):
    """将 GSEMAgent（FindStartTwoStage + 图遍历）接入 TTL 检索接口。

    每次 graph_state 中经验数量变化时自动重新初始化 FindStartTwoStage，
    确保新增的 TTL 经验能被检索到。embedding 增量通过缓存文件自动补全。
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._finder = None
        self._last_exp_count: int = 0

    def retrieve(
        self,
        case: Dict[str, Any],
        graph_state: "GraphState",
        top_k: int = 5,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        import sys
        from pathlib import Path
        from src.shared.logger import logger

        # 将 src/phase3/retrieval/ 加入 sys.path（retriver.py 内部有相对导入）
        _retrieval_dir = str(Path(__file__).resolve().parents[1] / "retrieval")
        if _retrieval_dir not in sys.path:
            sys.path.insert(0, _retrieval_dir)

        from find_start_two_stage import FindStartTwoStage
        from retriver import retrieve as _graph_retrieve

        # 经验数量变化时重新初始化 finder（含增量 embedding 生成）
        current_count = len(graph_state.experiences)
        if self._finder is None or current_count != self._last_exp_count:
            logger.log_info(
                f"  [GSEMRetrieval] 初始化检索器（经验数: {current_count}）…"
            )
            self._finder = FindStartTwoStage(
                experiences=dict(graph_state.experiences),
                top_k=top_k or self.top_k,
            )
            self._last_exp_count = current_count

        try:
            exps = _graph_retrieve(case, self._finder)
            # List[Dict] → List[Tuple[str, Dict]]
            return [(exp.get("id", ""), exp) for exp in exps if exp.get("id")]
        except Exception as e:
            logger.log_warning(f"  [GSEMRetrieval] 检索失败，返回空列表: {e}")
            return []
