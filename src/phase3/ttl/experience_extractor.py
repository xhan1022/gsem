"""TTL Step 3: Online Experience Extraction.

Extracts exactly one experience from a single online inference trajectory.
  - Correct inference  → Indication  (reuses Phase 1 INDICATION prompts)
  - Wrong  inference   → Contraindication (simplified single-trajectory prompt)
"""
import json
import re
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.shared.config import config
from src.shared.logger import logger
from src.phase1.prompts import INDICATION_SYSTEM_PROMPT, INDICATION_HUMAN_PROMPT
from .prompts import ONLINE_CONTRAINDICATION_SYSTEM_PROMPT, ONLINE_CONTRAINDICATION_HUMAN_PROMPT


class OnlineExperienceExtractor:
    """Step 3: Extract one experience from a single online trajectory."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )
        self.indication_prompt = ChatPromptTemplate.from_messages([
            ("system", INDICATION_SYSTEM_PROMPT),
            ("human", INDICATION_HUMAN_PROMPT),
        ])
        self.contraindication_prompt = ChatPromptTemplate.from_messages([
            ("system", ONLINE_CONTRAINDICATION_SYSTEM_PROMPT),
            ("human", ONLINE_CONTRAINDICATION_HUMAN_PROMPT),
        ])

    def extract(
        self,
        case: Dict[str, Any],
        trajectory: Dict[str, Any],
        is_correct: bool,
    ) -> Optional[Dict[str, Any]]:
        """Extract one experience from a single trajectory.

        Args:
            case: Clinical case
            trajectory: Inference trajectory from Step 1
            is_correct: Whether the inference was correct (Step 2 result)

        Returns:
            Experience dict, or None if extraction fails.
        """
        try:
            if is_correct:
                return self._extract_indication(case, trajectory)
            else:
                return self._extract_contraindication(case, trajectory)
        except Exception as e:
            logger.log_warning(f"[TTL Step 3] Experience extraction failed: {e}")
            return None

    # ------------------------------------------------------------------

    def _extract_indication(
        self,
        case: Dict[str, Any],
        trajectory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        trajectory_text = trajectory.get("trajectory_text", "")
        if not trajectory_text.strip():
            return None

        prompt = self.indication_prompt.format_messages(
            trajectory=trajectory_text,
            case_info=json.dumps(case, ensure_ascii=False, indent=2),
        )
        response = self.llm.invoke(prompt)
        parsed = self._parse_json(response.content)
        if not parsed:
            return None

        exp = parsed[0] if isinstance(parsed, list) else parsed
        if not isinstance(exp, dict) or not exp.get("content") or not exp.get("condition"):
            return None

        return {
            "type": "Indication",
            "task_type": case.get("task_type", ""),
            "condition": exp.get("condition", ""),
            "content": exp.get("content", ""),
            "evidence": exp.get("evidence", ""),
            "case_id": case.get("case_id", ""),
            "source": "ttl",
        }

    def _extract_contraindication(
        self,
        case: Dict[str, Any],
        trajectory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        trajectory_text = trajectory.get("trajectory_text", "")
        if not trajectory_text.strip():
            return None

        prompt = self.contraindication_prompt.format_messages(
            case_info=json.dumps(case, ensure_ascii=False, indent=2),
            failure_trajectory=trajectory_text,
            wrong_answer=trajectory.get("final_answer", ""),
            gold_answer=case.get("answer", ""),
        )
        response = self.llm.invoke(prompt)
        parsed = self._parse_json(response.content)
        if not parsed:
            return None

        exp = parsed[0] if isinstance(parsed, list) else parsed
        if not isinstance(exp, dict) or not exp.get("content") or not exp.get("condition"):
            return None

        return {
            "type": "Contraindication",
            "task_type": case.get("task_type", ""),
            "condition": exp.get("condition", ""),
            "content": exp.get("content", ""),
            "evidence": exp.get("evidence", ""),
            "case_id": case.get("case_id", ""),
            "source": "ttl",
        }

    @staticmethod
    def _parse_json(content: str) -> Any:
        """Try to extract JSON from LLM response (handles markdown code blocks)."""
        m = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"(\[.*\]|\{.*\})", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return None
