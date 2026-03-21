"""Stage 5: Cross-case Deduplication using LLM."""
import os
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.shared.config import config
from src.shared.logger import logger
from src.shared.utils import save_json, save_jsonl


# ============================================================================
# LLM-based Deduplication Prompts
# ============================================================================

DEDUPLICATION_SYSTEM_PROMPT = """You are a medical knowledge base curator specializing in clinical reasoning experience deduplication.

Your task is to deduplicate a list of clinical experiences by identifying and merging semantically similar or redundant entries.

DEDUPLICATION CRITERIA:
1. Two experiences are considered duplicates if:
   - Their "condition" (applicability context) is semantically equivalent
   - Their "content" (core guidance) conveys the same clinical reasoning pattern
   - They describe the same clinical decision-making scenario

2. When merging duplicates:
   - Keep the version with clearer, more precise wording
   - Keep the version with more complete information
   - If both are equally good, keep the first one

3. Preserve diversity:
   - Keep experiences that address different clinical contexts
   - Keep experiences with distinct reasoning patterns
   - Keep experiences with non-overlapping applicability conditions

OUTPUT FORMAT:
Return a JSON array containing only the deduplicated experiences. Each experience should retain all its original fields:
- type
- content
- condition
- task_type
- evidence

CRITICAL RULES:
- Output ONLY valid JSON array, no markdown, no explanation, no code blocks
- Start with [ and end with ]
- Each experience must be a complete JSON object with all original fields preserved
- Do not modify the content of kept experiences - copy them exactly as-is
"""

DEDUPLICATION_HUMAN_PROMPT = """Experience type: {experience_type}

Experiences to deduplicate ({count} total):
{experiences_json}

Return the deduplicated list as a JSON array."""


class Stage5Deduplication:
    """Stage 5: Deduplicate experiences across cases using LLM."""

    def __init__(self):
        # Initialize LLM for deduplication
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=0.0,  # Use deterministic output for deduplication
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

    def deduplicate_with_llm(self, experiences: List[Dict[str, Any]], exp_type: str) -> List[Dict[str, Any]]:
        """Deduplicate experiences using LLM.

        Args:
            experiences: List of experiences to deduplicate
            exp_type: Experience type (Indication or Contraindication)

        Returns:
            Deduplicated list of experiences
        """
        if not experiences:
            return []

        # If only one experience, no need to deduplicate
        if len(experiences) == 1:
            return experiences

        try:
            # Format experiences as JSON
            experiences_json = json.dumps(experiences, ensure_ascii=False, indent=2)

            # Create messages
            messages = [
                SystemMessage(content=DEDUPLICATION_SYSTEM_PROMPT),
                HumanMessage(content=DEDUPLICATION_HUMAN_PROMPT.format(
                    experience_type=exp_type,
                    count=len(experiences),
                    experiences_json=experiences_json
                ))
            ]

            # Call LLM
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Log response for debugging
            logger.log_info(f"    LLM 去重响应长度: {len(content)} 字符")

            # Extract JSON from response
            first_bracket = content.find('[')
            last_bracket = content.rfind(']')

            if first_bracket == -1 or last_bracket == -1 or last_bracket <= first_bracket:
                logger.log_warning(f"    无法找到有效的 JSON 数组，保留所有经验")
                return experiences

            json_str = content[first_bracket:last_bracket + 1]

            # Parse JSON
            deduplicated = json.loads(json_str)

            if not isinstance(deduplicated, list):
                logger.log_warning(f"    返回的不是数组，保留所有经验")
                return experiences

            logger.log_info(f"    {exp_type}: {len(experiences)} → {len(deduplicated)} (去重 {len(experiences) - len(deduplicated)} 条)")

            return deduplicated

        except json.JSONDecodeError as e:
            logger.log_warning(f"    JSON 解析失败: {str(e)}，保留所有经验")
            return experiences
        except Exception as e:
            logger.log_warning(f"    去重失败: {str(e)}，保留所有经验")
            return experiences

    def deduplicate_case_experiences(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate experiences for a single case (per-case deduplication).

        This method is called by the pipeline for each case individually.

        Args:
            experiences: List of experiences from one case (mixed Indication and Contraindication)

        Returns:
            Deduplicated experiences
        """
        if not experiences:
            return []

        # Separate by type
        indications = [exp for exp in experiences if exp.get("type") == "Indication"]
        contraindications = [exp for exp in experiences if exp.get("type") == "Contraindication"]

        # Deduplicate each type separately
        dedup_indications = self.deduplicate_with_llm(indications, "Indication") if indications else []
        dedup_contraindications = self.deduplicate_with_llm(contraindications, "Contraindication") if contraindications else []

        # Combine results
        return dedup_indications + dedup_contraindications

    def run(self, stage3_results: List[Dict[str, Any]], stage4_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect all experiences from all cases (no global deduplication).

        Note: Deduplication is done per-case in the pipeline.
        This method only collects and saves all experiences.

        Args:
            stage3_results: Results from Stage 3 (Indications, already deduplicated per-case)
            stage4_results: Results from Stage 4 (Contraindications, already deduplicated per-case)

        Returns:
            Experience library
        """
        logger.log_info("\n收集所有cases的经验...")

        # Collect all experiences (already deduplicated per-case)
        all_indications = []
        all_contraindications = []

        for result in stage3_results:
            all_indications.extend(result.get("indications", []))

        for result in stage4_results:
            all_contraindications.extend(result.get("contraindications", []))

        logger.log_info(f"  收集到 Indication: {len(all_indications)} 条 (已完成 per-case 去重)")
        logger.log_info(f"  收集到 Contraindication: {len(all_contraindications)} 条 (已完成 per-case 去重)")

        # Create final experience library (no global deduplication)
        experience_library = {
            "indications": all_indications,
            "contraindications": all_contraindications,
            "statistics": {
                "total_experiences": len(all_indications) + len(all_contraindications),
                "indication_count": len(all_indications),
                "contraindication_count": len(all_contraindications)
            }
        }

        # Save experience library (complete)
        output_path = os.path.join(config.paths.experiences_dir, "experience_library.json")
        save_json(experience_library, output_path)

        # Save all experiences as JSONL
        all_experiences = all_indications + all_contraindications
        jsonl_path = os.path.join(config.paths.experiences_dir, "experiences.jsonl")
        save_jsonl(all_experiences, jsonl_path)

        logger.log_info(f"\n最终经验库已保存到: {output_path}")

        return experience_library
