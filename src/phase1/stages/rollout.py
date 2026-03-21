"""Stage 1: Rollout - Trajectory Generation and Evaluation."""
import os
import json
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..agents import ReActAgent
from src.shared.config import config
from src.shared.logger import logger
from src.shared.utils import save_json
from ..prompt_provider import EVALUATION_SYSTEM_PROMPT, EVALUATION_HUMAN_PROMPT


class Stage1Rollout:
    """Stage 1: Generate trajectories and immediately evaluate them."""

    def __init__(self):
        self.agent = ReActAgent()
        self.sampling_count = config.pipeline.sampling_count

        # Evaluation LLM
        self.eval_llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATION_SYSTEM_PROMPT),
            ("human", EVALUATION_HUMAN_PROMPT)
        ])

    def evaluate_trajectory(self, trajectory: Dict[str, Any], answer: str) -> Dict[str, Any]:
        """Evaluate a single trajectory immediately after generation.

        Args:
            trajectory: Generated trajectory with final_answer
            answer: Gold standard answer

        Returns:
            Evaluation result
        """
        try:
            final_answer = trajectory.get("final_answer", "")
            total_steps = trajectory.get("total_steps", 0)

            # Check if final answer exists
            if not final_answer or not final_answer.strip():
                logger.log_warning(f"    Final answer is empty or too short")
                return {
                    "success": False,
                    "final_answer": "",
                    "match_reason": "No final answer found",
                    "score": 0
                }

            # Prepare evaluation input
            prompt = self.evaluation_prompt.format_messages(
                total_steps=total_steps,
                final_answer=final_answer,
                gold_standard=answer
            )

            response = self.eval_llm.invoke(prompt)
            content = response.content.strip()

            # Check if response is empty
            if not content:
                logger.log_warning(f"    LLM returned empty response for evaluation")
                return {
                    "success": False,
                    "final_answer": final_answer,
                    "match_reason": "LLM returned empty response",
                    "score": 0
                }

            # Log response for debugging
            logger.log_info(f"    评估响应: {content[:200]}...")

            # Extract JSON - find first { and last }
            first_brace = content.find('{')
            last_brace = content.rfind('}')

            if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
                logger.log_warning(f"    Cannot find valid JSON braces")
                logger.log_warning(f"    Full response: {content}")
                return {
                    "success": False,
                    "final_answer": final_answer,
                    "match_reason": "Invalid JSON format",
                    "score": 0
                }

            json_str = content[first_brace:last_brace + 1]

            # Log extracted JSON for debugging
            logger.log_info(f"    提取的JSON: {json_str[:200]}...")

            # Try to parse JSON
            evaluation = json.loads(json_str)

            return {
                "success": evaluation.get("success", False),
                "final_answer": evaluation.get("final_answer", final_answer),
                "match_reason": evaluation.get("match_reason", ""),
                "score": 100 if evaluation.get("success", False) else 0
            }

        except json.JSONDecodeError as e:
            logger.log_warning(f"    JSON parsing failed: {str(e)}")
            logger.log_warning(f"    Attempted to parse: {json_str if 'json_str' in locals() else 'N/A'}")
            return {
                "success": False,
                "final_answer": trajectory.get("final_answer", ""),
                "match_reason": f"JSON decode error: {str(e)}",
                "score": 0
            }
        except Exception as e:
            logger.log_warning(f"    Evaluation failed: {str(e)}")
            return {
                "success": False,
                "final_answer": trajectory.get("final_answer", ""),
                "match_reason": f"Evaluation error: {str(e)}",
                "score": 0
            }

    def process_case(self, case: Dict[str, Any], case_num: int, total_cases: int) -> Dict[str, Any]:
        """Process a single case: generate trajectories and evaluate immediately.

        Args:
            case: Clinical case data
            case_num: Current case number
            total_cases: Total number of cases

        Returns:
            Case with trajectories and evaluations
        """
        case_id = case.get("id", case.get("case_id", f"case_{case_num}"))
        answer = case.get("answer", "")
        logger.start_case(case_id, case_num)

        trajectories = []
        success_count = 0
        failure_count = 0

        for i in range(1, self.sampling_count + 1):
            try:
                # Generate trajectory
                trajectory = self.agent.generate_trajectory(case)

                # Immediately evaluate
                evaluation = self.evaluate_trajectory(trajectory, answer)

                # Add evaluation to trajectory
                trajectory["success"] = evaluation["success"]
                trajectory["evaluation"] = evaluation

                trajectories.append(trajectory)

                # Log with correct success/failure indicator
                if evaluation["success"]:
                    success_count += 1
                    logger.log_sampling(i, self.sampling_count, "生成中...", success=True)
                else:
                    failure_count += 1
                    logger.log_sampling(i, self.sampling_count, "生成中...", success=False)

            except Exception as e:
                logger.log_sampling(i, self.sampling_count, f"失败: {str(e)}", success=False)
                failure_count += 1

        result = {
            "case_id": case_id,
            "case": case,
            "trajectories": trajectories,
            "sampling_count": len(trajectories),
            "success_count": success_count,
            "failure_count": failure_count
        }

        # Save rollout results
        output_path = os.path.join(
            config.paths.trajectories_dir,
            f"{case_id}_rollout.json"
        )
        save_json(result, output_path)

        return result

    def run(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run Stage 1 for all cases.

        Args:
            cases: List of clinical cases

        Returns:
            List of cases with trajectories and evaluations
        """
        logger.log_stage("阶段1: Rollout（生成+评估）", "开始")
        results = []

        total_cases = len(cases)
        for idx, case in enumerate(cases, 1):
            try:
                result = self.process_case(case, idx, total_cases)
                results.append(result)
            except Exception as e:
                logger.log_error(f"Case processing failed: {str(e)}", case.get("case_id"))

        logger.log_stage("阶段1: Rollout（生成+评估）", "完成")
        return results
