"""Stage 6: Experience Replay Validation (ERV).

This stage validates the quality of deduplicated experience library using
an absolute performance threshold (τ) rather than relative baseline comparison.
"""
import json
import math
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.shared.config import config
from src.shared.logger import logger
from ..prompt_provider import (
    REACT_WITH_EXPERIENCES_SYSTEM_PROMPT,
    ERV_EXPERIENCES_SECTION,
    REACT_HUMAN_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_HUMAN_PROMPT
)


class Stage6ERV:
    """Stage 6: Experience Replay Validation (Absolute Threshold).

    New validation approach using absolute performance threshold:
    1. Replay sampling: re-sample all cases with deduplicated experience library
    2. Calculate quality metric: Δr = r_replay - τ (where τ is acceptable threshold)
    3. Map Δr to quality score using sigmoid function
    4. Attach quality field to all experiences

    Logic:
    - If r_replay > τ: experience helps reach acceptable performance (Quality > 50%)
    - If r_replay < τ: experience fails to meet threshold (Quality < 50%)
    - No dependency on baseline performance from Stage 1
    """

    def __init__(self):
        """Initialize ERV stage."""
        self.sampling_count = config.pipeline.sampling_count
        self.sigmoid_k = 1.0  # Fixed k=1 for simplicity (removed hyperparameter)
        self.threshold = config.pipeline.erv_threshold  # τ - acceptable performance threshold

        # LLM for trajectory generation
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=config.deepseek.temperature,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

        # LLM for evaluation
        self.eval_llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

    def format_experiences_for_prompt(self, experiences: List[Dict[str, Any]]) -> str:
        """Format experiences for inclusion in prompt.

        Args:
            experiences: List of experiences for a case

        Returns:
            Formatted string with all experiences
        """
        formatted_parts = []

        for idx, exp in enumerate(experiences, 1):
            exp_type = exp.get("type", "Unknown")
            task_type = exp.get("task_type", "")
            condition = exp.get("condition", "")
            content = exp.get("content", "")

            formatted_parts.append(
                f"Experience {idx} [{exp_type}]:\n"
                f"Task: {task_type}\n"
                f"Condition: {condition}\n"
                f"Guidance: {content}\n"
            )

        return "\n".join(formatted_parts)

    def generate_trajectory(
        self,
        case: Dict[str, Any],
        experiences_text: str = ""
    ) -> Dict[str, Any]:
        """Generate a single trajectory for a case.

        Args:
            case: Clinical case data
            experiences_text: Formatted experiences text (empty string for no experiences)

        Returns:
            Trajectory with reasoning steps
        """
        try:
            # Extract case information
            case_description = case.get("task", case.get("description", ""))
            task_type = case.get("task_type", "diagnosis")

            # Construct task instruction based on task_type
            task_instructions = {
                "diagnosis": "Based on the following clinical case, provide the most likely diagnosis.",
                "treatment": "Based on the following clinical case, provide the most appropriate treatment plan."
            }
            task_instruction = task_instructions.get(task_type, "Analyze the following clinical case.")

            # Construct full prompt with task instruction
            full_prompt = f"{task_instruction}\n\n{case_description}\n\nPlease use the ReAct method for reasoning."

            # Build system prompt with or without experiences
            if experiences_text:
                experiences_section = ERV_EXPERIENCES_SECTION.format(
                    experiences_text=experiences_text
                )
            else:
                experiences_section = ""

            system_prompt = REACT_WITH_EXPERIENCES_SYSTEM_PROMPT.format(
                experiences_section=experiences_section
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=full_prompt)
            ]

            # Generate response
            response = self.llm.invoke(messages)
            trajectory_text = response.content

            # Extract final answer (simple extraction)
            final_answer = ""
            if "Final Answer:" in trajectory_text:
                final_answer = trajectory_text.split("Final Answer:")[-1].strip()

            return {
                "trajectory_text": trajectory_text,
                "final_answer": final_answer
            }

        except Exception as e:
            logger.log_error(f"Failed to generate trajectory: {str(e)}", case.get("case_id"))
            raise

    def evaluate_trajectory(
        self,
        trajectory: Dict[str, Any],
        gold_answer: str
    ) -> bool:
        """Evaluate if trajectory's final answer matches gold standard.

        Args:
            trajectory: Generated trajectory
            gold_answer: Gold standard answer

        Returns:
            True if successful, False otherwise
        """
        try:
            final_answer = trajectory.get("final_answer", "")

            if not final_answer or not final_answer.strip():
                return False

            # Prepare evaluation prompt
            messages = [
                SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
                HumanMessage(content=EVALUATION_HUMAN_PROMPT.format(
                    total_steps=0,  # Not needed for ERV
                    final_answer=final_answer,
                    gold_standard=gold_answer
                ))
            ]

            response = self.eval_llm.invoke(messages)
            content = response.content.strip()

            # Extract JSON
            first_brace = content.find('{')
            last_brace = content.rfind('}')

            if first_brace == -1 or last_brace == -1:
                return False

            json_str = content[first_brace:last_brace + 1]
            evaluation = json.loads(json_str)

            return evaluation.get("success", False)

        except Exception as e:
            logger.log_warning(f"Evaluation failed: {str(e)}")
            return False

    def replay_sampling(
        self,
        case: Dict[str, Any],
        experiences: List[Dict[str, Any]]
    ) -> float:
        """Perform replay sampling with experiences.

        Args:
            case: Clinical case
            experiences: List of experiences for this case

        Returns:
            Replay success rate (0.0 to 1.0)
        """
        # Format experiences ONCE before the loop
        experiences_text = self.format_experiences_for_prompt(experiences)

        success_count = 0
        gold_answer = case.get("answer", "")

        for _ in range(self.sampling_count):
            trajectory = self.generate_trajectory(
                case,
                experiences_text=experiences_text  # Pass formatted text
            )
            is_success = self.evaluate_trajectory(trajectory, gold_answer)

            if is_success:
                success_count += 1

        return success_count / self.sampling_count

    def attach_quality_to_experiences(
        self,
        experiences: List[Dict[str, Any]],
        quality: float
    ) -> List[Dict[str, Any]]:
        """Attach quality score to all experiences in the list.

        Args:
            experiences: List of experiences
            quality: Quality score to attach

        Returns:
            List of experiences with quality field
        """
        for exp in experiences:
            exp["quality"] = quality

        return experiences

    def map_delta_to_quality(self, delta_r: float) -> float:
        """Map delta to quality score using sigmoid function with k=1.

        Args:
            delta_r: Deviation from threshold (Δr = r_replay - τ)

        Returns:
            Quality score in (0, 1) using sigmoid mapping

        Mapping formula:
            Q = sigmoid(delta_r) = 1 / (1 + exp(-delta_r))

        Properties:
            - delta_r = 0  → Q = 0.5 (at threshold)
            - delta_r > 0  → Q > 0.5 (above threshold, trustworthy)
            - delta_r < 0  → Q < 0.5 (below threshold, not trustworthy)
            - k = 1 (fixed, removed hyperparameter)
        """
        return 1.0 / (1.0 + math.exp(-self.sigmoid_k * delta_r))

    def validate_case_experiences(
        self,
        case: Dict[str, Any],
        experiences: List[Dict[str, Any]],
        baseline_rate: float = None  # Kept for compatibility but not used
    ) -> Dict[str, Any]:
        """Validate experiences for a single case using absolute threshold.

        New logic: Δr = r_replay - τ (where τ is the acceptable performance threshold)

        Args:
            case: Clinical case
            experiences: List of experiences for this case
            baseline_rate: [DEPRECATED] Not used, kept for compatibility

        Returns:
            Validation result with quality score
        """
        case_id = case.get("case_id", "unknown")

        logger.log_info(f"  [ERV验证] 经验数量: {len(experiences)}")
        logger.log_info(f"  [ERV验证] 可接受阈值 (τ): {self.threshold:.1%}")

        # Step 1: Replay sampling (with experiences)
        logger.log_info(f"  [ERV验证] 执行Replay采样 (m={self.sampling_count})...")
        r_replay = self.replay_sampling(case, experiences)
        logger.log_info(f"  [ERV验证] Replay正确率: {r_replay:.2%}")

        # Step 2: Calculate delta using absolute threshold (NEW LOGIC)
        delta_r = r_replay - self.threshold
        status = "可信" if delta_r >= 0 else "不可信"
        logger.log_info(f"  [ERV验证] 偏离阈值: Δr = {delta_r:+.2%} [{status}]")

        # Step 3: Map to quality
        quality = self.map_delta_to_quality(delta_r)
        logger.log_info(f"  [ERV验证] Quality映射: {quality:.4f}")

        return {
            "case_id": case_id,
            "threshold": self.threshold,
            "r_replay": r_replay,
            "delta_r": delta_r,
            "quality": quality,
            "experience_count": len(experiences)
        }

    def validate_experience_library(
        self,
        experience_library: Dict[str, Any],
        stage1_results: List[Dict[str, Any]],
        cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate deduplicated experience library per-case using absolute threshold.

        New logic: Δr = r_replay - τ (where τ is the acceptable performance threshold)
        - If r_replay > τ: experience is trustworthy (Quality > 50%)
        - If r_replay < τ: experience is not trustworthy (Quality < 50%)

        Args:
            experience_library: Deduplicated experience library from Stage 5
            stage1_results: Not used in new logic (kept for compatibility)
            cases: All clinical cases

        Returns:
            Experience library with quality scores attached to all experiences
        """
        logger.log_info("\\n" + "=" * 80)
        logger.log_info("开始ERV验证（基于绝对阈值 τ）")
        logger.log_info("=" * 80)

        # Collect all experiences from library
        all_experiences = []
        all_experiences.extend(experience_library.get("indications", []))
        all_experiences.extend(experience_library.get("contraindications", []))

        total_experiences = len(all_experiences)
        logger.log_info(f"\\n去重后的经验库大小:")
        logger.log_info(f"  Indication: {len(experience_library.get('indications', []))} 条")
        logger.log_info(f"  Contraindication: {len(experience_library.get('contraindications', []))} 条")
        logger.log_info(f"  总计: {total_experiences} 条")
        logger.log_info(f"\\n可接受表现阈值 (τ): {self.threshold:.1%}")

        if total_experiences == 0:
            logger.log_warning("经验库为空，跳过ERV验证")
            return experience_library

        # Validate per-case and collect delta_r values
        logger.log_info(f"\\n开始Per-Case验证（使用去重后经验库，m={self.sampling_count}）:")

        delta_r_list = []
        replay_rates = []

        for idx, case in enumerate(cases, 1):
            case_id = case.get("case_id", "unknown")

            # Replay sampling for this case with deduplicated experiences
            r_replay = self.replay_sampling(case, all_experiences)
            replay_rates.append(r_replay)

            # Calculate delta using absolute threshold (NEW LOGIC)
            delta_r = r_replay - self.threshold
            delta_r_list.append(delta_r)

            status = "可信" if delta_r >= 0 else "不可信"
            logger.log_info(f"  [{idx}/{len(cases)}] {case_id}: "
                          f"Replay={r_replay:.2%}, Δr={delta_r:+.2%} [{status}]")

        # Calculate average delta_r and replay rate
        avg_delta_r = sum(delta_r_list) / len(delta_r_list) if delta_r_list else 0.0
        avg_replay_rate = sum(replay_rates) / len(replay_rates) if replay_rates else 0.0

        logger.log_info(f"\\n平均回放准确率: {avg_replay_rate:.2%}")
        logger.log_info(f"平均偏离阈值: Δr_avg = {avg_delta_r:+.2%}")

        # Map average delta to quality
        quality = self.map_delta_to_quality(avg_delta_r)
        logger.log_info(f"Quality映射: {quality:.4f}")

        # Attach quality to all experiences
        logger.log_info(f"\\n将quality={quality:.4f}附加到所有{total_experiences}条经验")
        self.attach_quality_to_experiences(
            experience_library.get("indications", []),
            quality
        )
        self.attach_quality_to_experiences(
            experience_library.get("contraindications", []),
            quality
        )

        # Update statistics
        if "statistics" not in experience_library:
            experience_library["statistics"] = {}

        experience_library["statistics"]["erv_validation"] = {
            "threshold": self.threshold,
            "avg_replay_rate": avg_replay_rate,
            "avg_delta_r": avg_delta_r,
            "quality": quality,
            "total_cases_tested": len(cases),
            "sampling_count_per_case": self.sampling_count,
            "per_case_replay_rates": replay_rates,
            "per_case_delta_r": delta_r_list
        }

        logger.log_info("\\n" + "=" * 80)
        logger.log_info("ERV验证完成")
        logger.log_info("=" * 80)

        return experience_library
