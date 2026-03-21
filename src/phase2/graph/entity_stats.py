"""Step 4: Global Entity Statistics and IDF Calculation."""
import math
from typing import Dict, Any, List
from collections import defaultdict

from src.shared.logger import logger


class EntityStats:
    """Calculate global entity statistics and IDF weights."""

    def __init__(self):
        self.N = 0  # Total number of experiences
        self.df = {}  # Document frequency: entity -> count
        self.idf = {}  # IDF weights: entity -> score
        self.entity_postings = {}  # Entity -> list of experience IDs

    def compute_stats(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute entity statistics from all experiences.

        Args:
            experiences: List of experiences with 'canonical_entities' and 'id' fields

        Returns:
            Dict with 'N', 'df', 'idf' fields
        """
        logger.log_info(f"Computing entity statistics for {len(experiences)} experiences...")

        self.N = len(experiences)
        df_counter = defaultdict(int)
        postings = defaultdict(list)

        # Count document frequency and build postings
        for exp in experiences:
            exp_id = exp.get("id", "")
            entities = exp.get("canonical_entities", [])

            # Use set to count each entity once per experience
            for entity in set(entities):
                df_counter[entity] += 1
                postings[entity].append(exp_id)

        # Convert to regular dicts
        self.df = dict(df_counter)

        # Sort postings lists for stability
        self.entity_postings = {
            entity: sorted(ids) for entity, ids in postings.items()
        }

        # Calculate smoothed IDF
        # idf(v) = log((N + 1) / (df(v) + 1)) + 1
        self.idf = {}
        for entity, freq in self.df.items():
            self.idf[entity] = math.log((self.N + 1) / (freq + 1)) + 1

        logger.log_info(f"  Total experiences: {self.N}")
        logger.log_info(f"  Unique entities: {len(self.df)}")
        logger.log_info(f"Entity statistics computed")

        return {
            "N": self.N,
            "df": self.df,
            "idf": self.idf
        }

    def get_postings(self) -> Dict[str, List[str]]:
        """Get entity postings (inverted index).

        Returns:
            Dict mapping entity -> list of experience IDs
        """
        return self.entity_postings

    def get_idf_weight(self, entity: str) -> float:
        """Get IDF weight for an entity.

        Args:
            entity: Entity name

        Returns:
            IDF weight (default 1.0 if not found)
        """
        return self.idf.get(entity, 1.0)
