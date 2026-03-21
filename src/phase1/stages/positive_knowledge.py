"""Stage 3: Positive Knowledge Extraction."""
import os
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.shared.config import config
from src.shared.logger import logger
from src.shared.utils import save_json
from ..prompt_provider import INDICATION_SYSTEM_PROMPT, INDICATION_HUMAN_PROMPT


class Stage3PositiveKnowledgeExtraction:
    """Stage 3: Extract Indication knowledge from successful trajectories."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", INDICATION_SYSTEM_PROMPT),
            ("human", INDICATION_HUMAN_PROMPT)
        ])

    def extract_from_case(self, trajectories: List[Dict[str, Any]], case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract 1-2 Indication knowledge from all successful trajectories of a case.

        Args:
            trajectories: List of successful normalized trajectories
            case: Case information

        Returns:
            List of 1-2 Indication experiences for the entire case
        """
        if not trajectories:
            return []

        # Get task_type from case (already standardized in input data)
        task_type = case.get("task_type", "")

        raw_response = None
        parsed_content = None

        try:
            # Combine all successful trajectories (use original, not normalized)
            combined_text = ""
            for idx, traj in enumerate(trajectories, 1):
                trajectory_text = traj.get("original_trajectory", "")  # Correct field name
                if trajectory_text:
                    combined_text += f"\n\nTrajectory {idx}:\n{trajectory_text}"

            # Check if combined text is empty
            if not combined_text or len(combined_text.strip()) < 10:
                logger.log_warning(f"All trajectories are empty or too short, skipping extraction")
                return []

            # Truncate if too long
            if len(combined_text) > 5000:
                combined_text = combined_text[:5000] + "...(后文省略)"

            prompt = self.extraction_prompt.format_messages(
                trajectory=combined_text,
                case_info=json.dumps(case, ensure_ascii=False, indent=2)
            )

            response = self.llm.invoke(prompt)
            raw_response = response.content
            content = raw_response.strip()

            # Check if response is empty
            if not content:
                logger.log_warning(f"LLM returned empty response for Indication extraction")
                return []

            # Try to extract JSON from markdown or text
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                parsed_content = json_match.group(1)
            else:
                # Try to find JSON array or object
                json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
                if json_match:
                    parsed_content = json_match.group(1)
                else:
                    parsed_content = content

            # Parse JSON response
            experiences = json.loads(parsed_content)

            # Ensure it's a list
            if not isinstance(experiences, list):
                experiences = [experiences]

            # Validate and add metadata in specified order
            validated_experiences = []
            for exp in experiences:
                if self._validate_indication(exp):
                    # Create ordered dict with specified field order
                    # Use task_type from case input, not from LLM response
                    ordered_exp = {
                        "type": "Indication",
                        "task_type": task_type,  # Use task_type from case
                        "condition": exp.get("condition", ""),
                        "content": exp.get("content", ""),
                        "evidence": exp.get("evidence", ""),
                        "case_id": trajectories[0].get("case_id")
                    }
                    validated_experiences.append(ordered_exp)

            return validated_experiences

        except Exception as e:
            logger.log_warning(f"Indication extraction failed: {str(e)}")
            return []

    def _validate_indication(self, experience: Dict[str, Any]) -> bool:
        """Validate Indication experience format.

        Args:
            experience: Experience to validate

        Returns:
            True if valid
        """
        # task_type is now taken from case input, not from LLM response
        required_fields = ["content", "condition", "evidence"]
        return all(field in experience and experience[field] for field in required_fields)

    def process_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single case to extract Indication knowledge.

        Args:
            case_data: Case with normalized trajectories from Stage 3

        Returns:
            Case with extracted Indication experiences (1-2 per case)
        """
        case_id = case_data.get("case_id")
        case = case_data.get("case", {})
        normalized_trajectories = case_data.get("normalized_trajectories", [])

        # Collect all successful trajectories
        success_trajectories = [t for t in normalized_trajectories if t.get("success", False)]

        # Extract 1-2 indications from all successful trajectories combined
        all_indications = self.extract_from_case(success_trajectories, case)

        result = {
            "case_id": case_id,
            "indications": all_indications,
            "count": len(all_indications)
        }

        # Save positive knowledge
        output_path = os.path.join(
            config.paths.positive_knowledge_dir,
            f"{case_id}_indications.json"
        )
        save_json(result, output_path)

        return result

    def run(self, cases_with_normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run Stage 4 for all cases.

        Args:
            cases_with_normalized: Cases with normalized trajectories from Stage 3

        Returns:
            Cases with extracted Indication experiences
        """
        logger.log_stage("阶段4: 正向知识提取", "开始")
        results = []

        for case_data in cases_with_normalized:
            try:
                result = self.process_case(case_data)
                results.append(result)
            except Exception as e:
                logger.log_error(f"Positive knowledge extraction failed: {str(e)}", case_data.get("case_id"))

        logger.log_stage("阶段4: 正向知识提取", "完成")
        return results
