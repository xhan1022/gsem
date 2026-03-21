"""Stage 4: Failure Analysis."""
import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.shared.config import config
from src.shared.logger import logger
from src.shared.utils import save_json
from ..prompt_provider import (
    DIVERGENCE_SYSTEM_PROMPT, DIVERGENCE_HUMAN_PROMPT,
    CONTRAINDICATION_SYSTEM_PROMPT, CONTRAINDICATION_HUMAN_PROMPT
)


class Stage4FailureAnalysis:
    """Stage 4: Analyze failures and extract Contraindication."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

        # Initialize embedding client
        self.embedding_client = OpenAI(
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url
        )
        self.embedding_model = config.embedding.model_name

        self.divergence_prompt = ChatPromptTemplate.from_messages([
            ("system", DIVERGENCE_SYSTEM_PROMPT),
            ("human", DIVERGENCE_HUMAN_PROMPT)
        ])

        self.contraindication_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTRAINDICATION_SYSTEM_PROMPT),
            ("human", CONTRAINDICATION_HUMAN_PROMPT)
        ])

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.log_warning(f"Embedding API call failed: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(1536)  # Default dimension for text-embedding-3-small

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_most_similar_success(self, failure_traj: Dict[str, Any], success_trajs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the most similar success trajectory to a failure trajectory using embedding similarity.

        Args:
            failure_traj: Failed trajectory
            success_trajs: List of successful trajectories

        Returns:
            Most similar success trajectory
        """
        if not success_trajs:
            return None

        # Get embedding for failure trajectory
        failure_text = failure_traj.get("normalized_trajectory", "")
        failure_embedding = self.get_embedding(failure_text)

        # Calculate similarity with all success trajectories
        max_similarity = -1
        most_similar = None

        for success_traj in success_trajs:
            success_text = success_traj.get("normalized_trajectory", "")
            success_embedding = self.get_embedding(success_text)

            similarity = self.cosine_similarity(failure_embedding, success_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = success_traj

        logger.log_info(f"    最相似成功轨迹相似度: {max_similarity:.3f}")
        return most_similar

    def find_divergence_point(self, failure_traj: Dict[str, Any], success_traj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the fatal divergence point between failure and success trajectories.

        Args:
            failure_traj: Failed trajectory
            success_traj: Successful trajectory

        Returns:
            Divergence point information
        """
        try:
            failure_text = failure_traj.get("normalized_trajectory", "")
            success_text = success_traj.get("normalized_trajectory", "")

            # Check if trajectories are empty
            if not failure_text or not success_text:
                logger.log_warning(f"Empty trajectory in divergence detection")
                return None

            # Truncate if too long
            if len(failure_text) > 2000:
                failure_text = failure_text[:2000] + "...(后文省略)"
            if len(success_text) > 2000:
                success_text = success_text[:2000] + "...(后文省略)"

            # Extract answers from evaluations
            gold_answer = success_traj.get("evaluation", {}).get("final_answer", "correct answer")
            wrong_answer = failure_traj.get("evaluation", {}).get("final_answer", "incorrect answer")

            prompt = self.divergence_prompt.format_messages(
                failure_trajectory=failure_text,
                success_trajectory=success_text,
                gold_answer=gold_answer,
                wrong_answer=wrong_answer
            )

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Check if response is empty
            if not content:
                logger.log_warning(f"LLM returned empty response for divergence detection")
                return None

            # Try to extract JSON
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

            divergence = json.loads(content)

            return divergence

        except Exception as e:
            logger.log_warning(f"Divergence detection failed: {str(e)}")
            if 'response' in locals() and hasattr(response, 'content'):
                logger.log_warning(f"Response content (first 200 chars): {response.content[:200]}")
            return None

    def extract_contraindication(self, divergence: Dict[str, Any], case: Dict[str, Any],
                                failure_traj: Dict[str, Any], success_traj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract Contraindication from divergence point using original trajectories.

        Args:
            divergence: Divergence point information
            case: Case information
            failure_traj: Failed trajectory (with original_trajectory)
            success_traj: Success trajectory (with original_trajectory)

        Returns:
            Contraindication experience
        """
        try:
            # Use original trajectories for detailed analysis
            failure_original = failure_traj.get("original_trajectory", "")
            success_original = success_traj.get("original_trajectory", "")

            # Truncate if too long
            if len(failure_original) > 3000:
                failure_original = failure_original[:3000] + "...(truncated)"
            if len(success_original) > 3000:
                success_original = success_original[:3000] + "...(truncated)"

            # Get reference_analysis from case for additional context
            reference_analysis = case.get("reference_analysis", "")

            prompt = self.contraindication_prompt.format_messages(
                divergence=json.dumps(divergence, ensure_ascii=False, indent=2),
                case_info=json.dumps(case, ensure_ascii=False, indent=2),
                failure_trajectory=failure_original,
                success_trajectory=success_original,
                reference_analysis=reference_analysis
            )

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Check if response is empty
            if not content:
                logger.log_warning(f"LLM returned empty response for Contraindication extraction")
                return None

            # Try to extract JSON
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

            contraindication = json.loads(content)

            if self._validate_contraindication(contraindication):
                return contraindication

        except Exception as e:
            logger.log_warning(f"Contraindication extraction failed: {str(e)}")
            if 'response' in locals() and hasattr(response, 'content'):
                logger.log_warning(f"Response content (first 200 chars): {response.content[:200]}")

        return None

    def _validate_contraindication(self, experience: Dict[str, Any]) -> bool:
        """Validate Contraindication format."""
        # task_type is now taken from case input, not from LLM response
        required_fields = ["content", "condition", "evidence"]
        return all(field in experience and experience[field] for field in required_fields)

    def process_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single case for failure analysis.

        Args:
            case_data: Case with normalized trajectories from Stage 3

        Returns:
            Case with extracted Contraindication
        """
        case_id = case_data.get("case_id")
        case = case_data.get("case", {})
        normalized_trajectories = case_data.get("normalized_trajectories", [])

        # Get task_type from case (already standardized in input data)
        task_type = case.get("task_type", "")

        # Separate success and failure trajectories
        success_trajs = [t for t in normalized_trajectories if t.get("success", False)]
        failure_trajs = [t for t in normalized_trajectories if not t.get("success", False)]

        k = len(failure_trajs)  # Number of failures
        contraindications = []

        # Pair each failure with the most similar success trajectory
        # Extract Contraindication for EACH pair
        num_pairs = 0
        for idx, failure_traj in enumerate(failure_trajs, 1):
            if success_trajs:
                logger.log_info(f"  处理失败轨迹 {idx}/{k}，寻找最相似成功轨迹...")

                # Find most similar success trajectory using embedding
                success_traj = self.find_most_similar_success(failure_traj, success_trajs)

                if success_traj:
                    divergence = self.find_divergence_point(failure_traj, success_traj)

                    if divergence:
                        num_pairs += 1

                        # Extract Contraindication for this pair (using original trajectories)
                        contraindication = self.extract_contraindication(divergence, case, failure_traj, success_traj)
                        if contraindication:
                            # Create ordered dict with specified field order
                            # Use task_type from case input, not from LLM response
                            ordered_contra = {
                                "type": "Contraindication",
                                "task_type": task_type,  # Use task_type from case
                                "condition": contraindication.get("condition", ""),
                                "content": contraindication.get("content", ""),
                                "evidence": contraindication.get("evidence", ""),
                                "case_id": case_id
                            }
                            contraindications.append(ordered_contra)

        logger.log_pairing(num_pairs)

        result = {
            "case_id": case_id,
            "k": k,
            "num_pairs": num_pairs,
            "contraindications": contraindications,
            "contraindication_count": len(contraindications)
        }

        # Save failure analysis
        output_path = os.path.join(
            config.paths.failure_analysis_dir,
            f"{case_id}_failure_analysis.json"
        )
        save_json(result, output_path)

        return result

    def run(self, cases_with_normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run Stage 5 for all cases.

        Args:
            cases_with_normalized: Cases with normalized trajectories from Stage 3

        Returns:
            Cases with failure analysis results
        """
        logger.log_stage("阶段5: 失败分析", "开始")
        results = []

        for case_data in cases_with_normalized:
            try:
                result = self.process_case(case_data)
                results.append(result)
            except Exception as e:
                logger.log_error(f"Failure analysis failed: {str(e)}", case_data.get("case_id"))

        logger.log_stage("阶段5: 失败分析", "完成")
        return results
