"""Stage 2: Trajectory Normalization."""
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.shared.config import config
from src.shared.logger import logger
from src.shared.utils import save_json
from ..prompt_provider import NORMALIZATION_SYSTEM_PROMPT, NORMALIZATION_HUMAN_PROMPT


class Stage2TrajectoryNormalization:
    """Stage 2: Normalize trajectories to 30-40% of original length."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

        self.normalization_prompt = ChatPromptTemplate.from_messages([
            ("system", NORMALIZATION_SYSTEM_PROMPT),
            ("human", NORMALIZATION_HUMAN_PROMPT)
        ])

    def normalize_trajectory(self, trajectory: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single trajectory.

        Args:
            trajectory: Original trajectory
            evaluation: Evaluation result

        Returns:
            Normalized trajectory
        """
        try:
            trajectory_text = trajectory.get("trajectory_text", "")
            original_length = len(trajectory_text)
            target_length = int(original_length * 0.35)  # Target 35% (middle of 30-40%)

            prompt = self.normalization_prompt.format_messages(
                trajectory=trajectory_text,
                original_length=original_length,
                target_length=target_length
            )

            response = self.llm.invoke(prompt)
            normalized_text = response.content

            normalized_length = len(normalized_text)
            compression_ratio = normalized_length / original_length if original_length > 0 else 0

            return {
                "case_id": trajectory.get("case_id"),
                "original_trajectory": trajectory_text,
                "normalized_trajectory": normalized_text,
                "original_length": original_length,
                "normalized_length": normalized_length,
                "compression_ratio": compression_ratio,
                "success": evaluation.get("success", False),
                "evaluation": evaluation
            }

        except Exception as e:
            logger.log_warning(f"Normalization failed: {str(e)}")
            # Return original if normalization fails
            return {
                "case_id": trajectory.get("case_id"),
                "original_trajectory": trajectory.get("trajectory_text", ""),
                "normalized_trajectory": trajectory.get("trajectory_text", ""),
                "original_length": len(trajectory.get("trajectory_text", "")),
                "normalized_length": len(trajectory.get("trajectory_text", "")),
                "compression_ratio": 1.0,
                "success": evaluation.get("success", False),
                "evaluation": evaluation
            }

    def process_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single case to normalize all trajectories.

        Args:
            case_data: Case with trajectories from Stage 1 (Rollout)

        Returns:
            Case with normalized trajectories
        """
        case_id = case_data.get("case_id")
        trajectories = case_data.get("trajectories", [])

        normalized_trajectories = []
        for trajectory in trajectories:
            # Extract evaluation from trajectory (it's embedded in each trajectory)
            evaluation = trajectory.get("evaluation", {})
            normalized = self.normalize_trajectory(trajectory, evaluation)
            normalized_trajectories.append(normalized)

        result = {
            "case_id": case_id,
            "case": case_data.get("case", {}),
            "normalized_trajectories": normalized_trajectories,
            "success_count": case_data.get("success_count", 0),
            "failure_count": case_data.get("failure_count", 0)
        }

        # Save normalized trajectories
        output_path = os.path.join(
            config.paths.normalized_dir,
            f"{case_id}_normalized.json"
        )
        save_json(result, output_path)

        return result

    def run(self, cases_with_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run Stage 3 for all cases.

        Args:
            cases_with_evaluations: Cases with evaluations from Stage 2

        Returns:
            Cases with normalized trajectories
        """
        logger.log_stage("阶段3: 轨迹标准化", "开始")
        results = []

        for case_data in cases_with_evaluations:
            try:
                result = self.process_case(case_data)
                results.append(result)
            except Exception as e:
                logger.log_error(f"Normalization failed: {str(e)}", case_data.get("case_id"))

        logger.log_stage("阶段3: 轨迹标准化", "完成")
        return results
