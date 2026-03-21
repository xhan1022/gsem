"""Step 3: Role-Edge Structure Extraction.

This step extracts decision-flow structure (reasoning skeleton) using role-to-role edges.

Allowed role-edges (fixed set of 6):
- Condition→Action: Core decision trigger
- Condition→Constraint: Condition introduces limitation/boundary
- Constraint→Action: Limitation changes/redirects action choice
- Action→Outcome: Action aims to achieve clinical goal
- Action→Rationale: Action supported by medical justification
- Condition→Outcome: Condition directly highlights target goal

Typically extracts 2-5 edges per experience.
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
    ROLE_EDGE_EXTRACTION_SYSTEM_PROMPT,
    ROLE_EDGE_EXTRACTION_HUMAN_PROMPT
)


class RoleEdgeExtractor:
    """Extract role-edge decision-flow structure from experiences."""

    # Fixed set of allowed role-edges
    ALLOWED_EDGES = {
        "Condition→Action",
        "Condition→Condition",
        "Condition→Constraint",
        "Constraint→Action",
        "Constraint→Rationale",
        "Constraint→Outcome",
        "Action→Outcome",
        "Action→Rationale",
        "Action→Constraint",
        "Condition→Outcome",
        "Condition→Rationale",
        "Rationale→Action",
        "Rationale→Outcome",
        "Rationale→Constraint",
        "Action→Action"
    }

    def __init__(self):
        """Initialize role-edge extractor."""
        self.llm = ChatOpenAI(
            model_name=config.deepseek.model_name,
            openai_api_base=config.deepseek.base_url,
            openai_api_key=config.deepseek.api_key,
            temperature=0.0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROLE_EDGE_EXTRACTION_SYSTEM_PROMPT),
            ("human", ROLE_EDGE_EXTRACTION_HUMAN_PROMPT)
        ])

        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

        self.stats = {
            "total_experiences": 0,
            "total_edges": 0,
            "edges_by_type": {edge: 0 for edge in self.ALLOWED_EDGES},
            "avg_edges_per_exp": 0.0,
            "invalid_edges": 0,
            "errors": 0
        }

    def extract_role_edges(
        self,
        core_entities: List[Dict],
        condition: str,
        content: str
    ) -> tuple[List[str], List[Dict]]:
        """Extract role-edges and entity-edges from a single experience.

        Args:
            core_entities: List of core entities with roles
            condition: Experience condition text
            content: Experience content text

        Returns:
            Tuple of (role_edges, entity_edges) where:
            - role_edges: List of role-edge strings (e.g., ["Condition→Action"])
            - entity_edges: List of entity-edge dicts with edge, from_entity, to_entity
        """
        try:
            # Format core_entities as JSON string for prompt
            core_entities_json = json.dumps(core_entities, ensure_ascii=False)

            response = self.chain.invoke({
                "core_entities_json": core_entities_json,
                "condition": condition,
                "content": content
            })

            # Parse role_edges from response
            role_edges = response.get("role_edges", [])
            entity_edges = response.get("entity_edges", [])

            # Validate role edges
            valid_role_edges = []
            for edge in role_edges:
                if edge in self.ALLOWED_EDGES:
                    valid_role_edges.append(edge)
                else:
                    logger.log_warning(f"Invalid edge (not in allowed set): {edge}")
                    self.stats["invalid_edges"] += 1

            # Deduplicate role edges
            unique_role_edges = list(dict.fromkeys(valid_role_edges))

            # Validate entity edges (ensure they match valid role edges)
            allowed_entities = {
                item.get("entity", "")
                for item in core_entities
                if isinstance(item, dict) and item.get("entity")
            }
            valid_entity_edges = []
            for e_edge in entity_edges:
                edge_type = e_edge.get("edge", "")
                from_entity = e_edge.get("from_entity", "")
                to_entity = e_edge.get("to_entity", "")
                if (
                    edge_type in self.ALLOWED_EDGES
                    and from_entity in allowed_entities
                    and to_entity in allowed_entities
                ):
                    valid_entity_edges.append(e_edge)

            return unique_role_edges, valid_entity_edges

        except Exception as e:
            logger.log_error(f"Error extracting role-edges: {e}")
            self.stats["errors"] += 1
            return [], []

    def _build_canonical_core_entities_with_roles(self, experience: Dict) -> List[Dict]:
        """Build role-tagged canonical entities for Step 3 prompting.

        `canonical_entities` stores normalized (lowercase) entity strings, sorted
        alphabetically — NOT position-aligned with `core_entities`. We match by
        normalizing each core entity name and looking it up in canonical_entities.
        """
        core_entities = experience.get("core_entities", [])
        canonical_entities = experience.get("canonical_entities", [])

        if not core_entities or not canonical_entities:
            return core_entities

        # Build lookup: normalized name → canonical name
        canon_set = set()
        for item in canonical_entities:
            name = item.get("entity") if isinstance(item, dict) else item
            if name:
                canon_set.add(name)

        canonical_core_entities = []
        for core_item in core_entities:
            if not isinstance(core_item, dict):
                continue
            role = core_item.get("role")
            raw_name = core_item.get("entity", "")
            if not role or not raw_name:
                continue
            # Match by normalizing the raw name (lowercase + collapse whitespace)
            normalized = " ".join(raw_name.strip().lower().split())
            canon_name = normalized if normalized in canon_set else raw_name
            canonical_core_entities.append({"entity": canon_name, "role": role})

        return canonical_core_entities if canonical_core_entities else core_entities

    def process_single(self, experience: Dict) -> Dict:
        """Process a single experience for role-edge extraction.

        Args:
            experience: Experience dict with 'core_entities', 'condition', 'content'

        Returns:
            Experience dict with added 'role_edges' and 'entity_edges'
        """
        core_entities = self._build_canonical_core_entities_with_roles(experience)
        condition = experience.get("condition", "")
        content = experience.get("content", "")

        # Extract role-edges and entity-edges
        role_edges, entity_edges = self.extract_role_edges(core_entities, condition, content)

        # Add to experience
        experience["role_edges"] = role_edges
        experience["entity_edges"] = entity_edges

        # Update stats
        self.stats["total_edges"] += len(role_edges)
        for edge in role_edges:
            if edge in self.stats["edges_by_type"]:
                self.stats["edges_by_type"][edge] += 1

        return experience

    def process_experiences(
        self,
        experiences: List[Dict],
        output_path: str = None,
        incremental: bool = True
    ) -> List[Dict]:
        """Process all experiences with optional incremental output.

        Args:
            experiences: List of experience dicts with core_entities
            output_path: Optional path for incremental JSONL output
            incremental: Whether to write results incrementally

        Returns:
            List of experiences with role_edges
        """
        logger.log_info("\n" + "=" * 80)
        logger.log_info("STEP 3: Role-Edge Structure Extraction")
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

            # Log progress for each experience
            self.stats["total_experiences"] += 1
            exp_id = result.get('id', f'exp_{idx}')
            role_edges = result.get('role_edges', [])
            edges_str = ', '.join(role_edges) if role_edges else 'No edges'
            logger.log_info(f"  [{idx + 1}/{len(experiences)}] {exp_id}: {edges_str}")

        # Close output file
        if output_file:
            output_file.close()

        # Compute final stats
        if self.stats["total_experiences"] > 0:
            self.stats["avg_edges_per_exp"] = (
                self.stats["total_edges"] / self.stats["total_experiences"]
            )

        # Log final stats
        logger.log_info("\n" + "=" * 80)
        logger.log_info("Step 3 Complete")
        logger.log_info("=" * 80)
        logger.log_info(f"  Total experiences: {self.stats['total_experiences']}")
        logger.log_info(f"  Total edges extracted: {self.stats['total_edges']}")
        logger.log_info(f"  Average edges per experience: {self.stats['avg_edges_per_exp']:.2f}")
        logger.log_info(f"  Edges by type:")
        for edge_type, count in self.stats["edges_by_type"].items():
            if count > 0:  # Only show edge types that were used
                logger.log_info(f"    {edge_type}: {count}")
        logger.log_info(f"  Invalid edges filtered: {self.stats['invalid_edges']}")
        logger.log_info(f"  Errors: {self.stats['errors']}")

        return results

    def save_results(self, experiences: List[Dict], output_path: str):
        """Save experiences with role-edges.

        Args:
            experiences: Experiences with role_edges
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for exp in experiences:
                f.write(json.dumps(exp, ensure_ascii=False) + '\n')

        logger.log_info(f"\nRole-edge extraction results saved to: {output_path}")
