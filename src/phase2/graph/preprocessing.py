"""Step 1: Preprocessing - ID Assignment and Task Type Normalization.

This step prepares experiences for entity extraction by:
- Assigning unique IDs to experiences
- Normalizing task types
"""
import json
import os
from typing import List, Dict

from src.shared.logger import logger


class ExperiencePreprocessor:
    """Preprocess experiences before entity extraction."""

    def __init__(self):
        """Initialize preprocessor."""
        self.stats = {
            "total_experiences": 0,
            "id_assigned": 0,
            "task_types_normalized": 0,
        }

    def assign_ids(self, experiences: List[Dict]) -> List[Dict]:
        """Assign unique IDs to experiences if not already present.

        Args:
            experiences: List of experience dicts

        Returns:
            List of experiences with 'id' field
        """
        logger.log_info("\nAssigning IDs to experiences...")

        for idx, exp in enumerate(experiences):
            if 'id' not in exp:
                # Generate unique ID for each experience
                exp['id'] = f"exp_{idx:04d}"
                self.stats['id_assigned'] += 1

            self.stats['total_experiences'] += 1

        logger.log_info(f"  Assigned IDs to {self.stats['id_assigned']} experiences")
        logger.log_info(f"  Total experiences: {self.stats['total_experiences']}")

        return experiences

    def normalize_task_types(self, experiences: List[Dict]) -> List[Dict]:
        """Normalize task_type field if present.

        Args:
            experiences: List of experience dicts

        Returns:
            List of experiences with normalized 'task_type_norm' field
        """
        logger.log_info("\nNormalizing task types...")

        # Task type normalization mapping (if needed)
        task_type_map = {
            # Add mappings if needed
            # "diagnosis": "diagnosis",
            # "treatment": "treatment",
        }

        for exp in experiences:
            if 'task_type' in exp:
                task_type = exp['task_type'].strip().lower()
                exp['task_type_norm'] = task_type_map.get(task_type, task_type)
                self.stats['task_types_normalized'] += 1
            else:
                exp['task_type_norm'] = 'unknown'

        logger.log_info(f"  Normalized task types: {self.stats['task_types_normalized']}")

        return experiences

    def process(self, experiences: List[Dict]) -> List[Dict]:
        """Run complete preprocessing pipeline.

        Args:
            experiences: List of experience dicts

        Returns:
            Preprocessed experiences
        """
        logger.log_info("\n" + "=" * 80)
        logger.log_info("STEP 1: Preprocessing (ID Assignment + Task Type Normalization)")
        logger.log_info("=" * 80)

        experiences = self.assign_ids(experiences)
        experiences = self.normalize_task_types(experiences)

        logger.log_info("\n" + "=" * 80)
        logger.log_info("Step 1 Complete")
        logger.log_info("=" * 80)
        logger.log_info(f"  Total experiences: {self.stats['total_experiences']}")
        logger.log_info(f"  IDs assigned: {self.stats['id_assigned']}")
        logger.log_info(f"  Task types normalized: {self.stats['task_types_normalized']}")

        return experiences

    def save_results(self, experiences: List[Dict], output_path: str):
        """Save preprocessed experiences.

        Args:
            experiences: Preprocessed experiences
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for exp in experiences:
                f.write(json.dumps(exp, ensure_ascii=False) + '\n')

        logger.log_info(f"\nPreprocessed experiences saved to: {output_path}")
