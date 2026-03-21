"""TTL Step 1: Inference with Retrieved Experiences.

Retrieves top-K experiences via the retrieval interface, injects them into the
system prompt, and executes one complete ReAct inference.
"""
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.shared.config import config
from src.shared.logger import logger
from src.phase1.prompts import REACT_WITH_EXPERIENCES_SYSTEM_PROMPT, ERV_EXPERIENCES_SECTION
from .retrieval_tool import BaseRetrievalInterface, StubRetrievalInterface


class ReasoningAgent:
    """Step 1: Run one inference with retrieved experiences injected into context."""

    def __init__(
        self,
        retrieval_interface: BaseRetrievalInterface = None,
        top_k: int = 5,
    ):
        self.retrieval = retrieval_interface or StubRetrievalInterface()
        self.top_k = top_k
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=config.deepseek.temperature,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

    def run(
        self,
        case: Dict[str, Any],
        graph_state: Any,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
        """Run one inference.

        Args:
            case: Clinical case dict
            graph_state: Current GraphState

        Returns:
            (trajectory_dict, retrieved_pairs)
            - trajectory_dict: {"trajectory_text", "final_answer", "case_id"}
            - retrieved_pairs: [(exp_id, exp_dict), ...] ordered by relevance
        """
        # Retrieve top-K experiences
        retrieved_pairs = self.retrieval.retrieve(case, graph_state, self.top_k)
        logger.log_info(f"  [Step 1] Retrieved {len(retrieved_pairs)} experiences")

        # Build system prompt with or without experiences
        experiences_text = self._format_experiences(retrieved_pairs)
        if experiences_text:
            experiences_section = ERV_EXPERIENCES_SECTION.format(
                experiences_text=experiences_text
            )
        else:
            experiences_section = ""

        system_prompt = REACT_WITH_EXPERIENCES_SYSTEM_PROMPT.format(
            experiences_section=experiences_section
        )

        # Build user message
        case_description = case.get("description", case.get("task", ""))
        task_type = case.get("task_type", "diagnosis")
        task_instructions = {
            "diagnosis": "Based on the following clinical case, provide the most likely diagnosis.",
            "treatment": "Based on the following clinical case, provide the most appropriate treatment plan.",
        }
        task_instruction = task_instructions.get(
            task_type, "Analyze the following clinical case."
        )
        full_prompt = (
            f"{task_instruction}\n\n{case_description}\n\n"
            "Please use the ReAct method for reasoning."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            trajectory_text = response.content
            final_answer = ""
            if "Final Answer:" in trajectory_text:
                final_answer = trajectory_text.split("Final Answer:")[-1].strip()
            trajectory = {
                "trajectory_text": trajectory_text,
                "final_answer": final_answer,
                "case_id": case.get("case_id", ""),
            }
        except Exception as e:
            logger.log_error(f"[TTL Step 1] Inference failed: {e}")
            trajectory = {
                "trajectory_text": "",
                "final_answer": "",
                "case_id": case.get("case_id", ""),
            }

        logger.log_info(
            f"  [Step 1] Done. Final answer: {trajectory['final_answer'][:80]!r}"
        )
        return trajectory, retrieved_pairs

    def _format_experiences(
        self, retrieved_pairs: List[Tuple[str, Dict[str, Any]]]
    ) -> str:
        if not retrieved_pairs:
            return ""
        parts = []
        for idx, (eid, exp) in enumerate(retrieved_pairs, 1):
            parts.append(
                f"Experience {idx} [{exp.get('type', 'Unknown')}]:\n"
                f"Task: {exp.get('task_type', '')}\n"
                f"Condition: {exp.get('condition', '')}\n"
                f"Guidance: {exp.get('content', '')}\n"
            )
        return "\n".join(parts)
