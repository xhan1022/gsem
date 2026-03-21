"""Step 2: Core Entity Extraction with Roles.

This step extracts compact decision-driving entities from experiences with fixed roles:
- Condition: 1-2 anchors determining strategy applicability
- Constraint: 0-1 anchor blocking/limiting default approach
- Action: 1-2 anchors truly changing decisions
- Rationale: 0-1 anchor explaining why strategy holds
- Outcome: 0-1 anchor describing intent/benefit

Typically extracts 5-8 entities per experience.
"""
import json
import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.shared.config import config
from src.shared.logger import logger
from .prompts import (
    CORE_ENTITY_EXTRACTION_SYSTEM_PROMPT,
    CORE_ENTITY_EXTRACTION_HUMAN_PROMPT
)


class CoreEntityExtractor:
    """Extract core entities with roles from clinical experiences."""

    def __init__(self):
        """Initialize core entity extractor."""
        self.llm = ChatOpenAI(
            model_name=config.deepseek.model_name,
            openai_api_base=config.deepseek.base_url,
            openai_api_key=config.deepseek.api_key,
            temperature=0.0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CORE_ENTITY_EXTRACTION_SYSTEM_PROMPT),
            ("human", CORE_ENTITY_EXTRACTION_HUMAN_PROMPT)
        ])

        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

        self.stats = {
            "total_experiences": 0,
            "total_entities": 0,
            "entities_by_role": {
                "Condition": 0,
                "Constraint": 0,
                "Action": 0,
                "Rationale": 0,
                "Outcome": 0
            },
            "avg_entities_per_exp": 0.0,
            "errors": 0
        }

    def extract_entities(self, condition: str, content: str) -> List[Dict]:
        """Extract core entities with roles from a single experience.

        Args:
            condition: Experience condition text
            content: Experience content text

        Returns:
            List of dicts with "entity" and "role" keys
        """
        try:
            response = self.chain.invoke({
                "condition": condition,
                "content": content
            })

            # Parse core_entities from response
            core_entities = response.get("core_entities", [])

            # Validate structure
            valid_entities = []
            for ent_dict in core_entities:
                if isinstance(ent_dict, dict) and "entity" in ent_dict and "role" in ent_dict:
                    valid_entities.append(ent_dict)
                else:
                    logger.log_warning(f"Invalid entity structure: {ent_dict}")

            return valid_entities

        except Exception as e:
            logger.log_error(f"Error extracting entities: {e}")
            import traceback
            logger.log_error(traceback.format_exc())
            self.stats["errors"] += 1
            return []

    def process_single(self, experience: Dict) -> Dict:
        """Process a single experience for core entity extraction.

        Args:
            experience: Experience dict with 'condition' and 'content'

        Returns:
            Experience dict with added 'core_entities' (no normalization yet)
        """
        condition = experience.get("condition", "")
        content = experience.get("content", "")

        # Extract core entities with roles
        core_entities = self.extract_entities(condition, content)

        # Add to experience (no normalization at this stage)
        experience["core_entities"] = core_entities

        # Update stats
        self.stats["total_entities"] += len(core_entities)
        for ent in core_entities:
            role = ent.get("role", "Unknown")
            if role in self.stats["entities_by_role"]:
                self.stats["entities_by_role"][role] += 1

        return experience

    def process_experiences(
        self,
        experiences: List[Dict],
        output_path: str = None,
        incremental: bool = True
    ) -> List[Dict]:
        """Process all experiences with optional incremental output.

        Args:
            experiences: List of experience dicts
            output_path: Optional path for incremental JSONL output
            incremental: Whether to write results incrementally

        Returns:
            List of experiences with core_entities and canonical_entities
        """
        logger.log_info("\n" + "=" * 80)
        logger.log_info("STEP 2: Core Entity Extraction (with Roles)")
        logger.log_info("=" * 80)
        logger.log_info(f"Processing {len(experiences)} experiences...")

        # Prepare incremental output file
        output_file = None
        if incremental and output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_file = open(output_path, 'w', encoding='utf-8')
            logger.log_info(f"Incremental output enabled: {output_path}")

        results = []
        for idx, exp in enumerate(experiences):
            # Process experience
            result = self.process_single(exp)
            results.append(result)

            # Incremental write
            if output_file:
                output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                output_file.flush()

            # Log progress (every record)
            self.stats["total_experiences"] += 1
            num_entities = len(result.get('core_entities', []))
            roles = [e['role'] for e in result.get('core_entities', [])]
            logger.log_info(f"  [{idx + 1}/{len(experiences)}] {result['id']}: {num_entities} entities ({', '.join(set(roles))})")

        # Close output file
        if output_file:
            output_file.close()

        # Compute final stats
        if self.stats["total_experiences"] > 0:
            self.stats["avg_entities_per_exp"] = (
                self.stats["total_entities"] / self.stats["total_experiences"]
            )

        # Log final stats
        logger.log_info("\n" + "=" * 80)
        logger.log_info("Step 2 Complete")
        logger.log_info("=" * 80)
        logger.log_info(f"  Total experiences: {self.stats['total_experiences']}")
        logger.log_info(f"  Total entities extracted: {self.stats['total_entities']}")
        logger.log_info(f"  Average entities per experience: {self.stats['avg_entities_per_exp']:.2f}")
        logger.log_info(f"  Entities by role:")
        for role, count in self.stats["entities_by_role"].items():
            logger.log_info(f"    {role}: {count}")
        logger.log_info(f"  Errors: {self.stats['errors']}")
        logger.log_info(f"\n  Note: Normalization will be done in Step 2.5")

        return results

    def save_results(self, experiences: List[Dict], output_path: str):
        """Save experiences with core entities.

        Args:
            experiences: Experiences with core_entities
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for exp in experiences:
                f.write(json.dumps(exp, ensure_ascii=False) + '\n')

        logger.log_info(f"\nCore entity extraction results saved to: {output_path}")
