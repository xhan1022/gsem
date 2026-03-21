"""Lexicon management for entity normalization."""
import json
import os
from typing import Dict, List, Set

from .normalizer import normalize_string


class Lexicon:
    """Self-maintained lexicon for entity normalization.

    Maintains two core data structures:
    - canonical_to_aliases: Maps canonical entity to its aliases
    - alias_to_canonical: Maps each alias to its canonical form

    Normalization strategy:
    - Hit: If normalized entity exists in alias_to_canonical, use its canonical
    - Miss: Create new canonical, add to both dictionaries
    """

    def __init__(self, lexicon_path: str = "data/graph_output/lexicon.json"):
        """Initialize lexicon.

        Args:
            lexicon_path: Path to lexicon JSON file
        """
        self.lexicon_path = lexicon_path
        self.canonical_to_aliases: Dict[str, List[str]] = {}
        self.alias_to_canonical: Dict[str, str] = {}

        # Load existing lexicon if available
        self.load()

    def load(self) -> None:
        """Load lexicon from disk if exists."""
        if not os.path.exists(self.lexicon_path):
            return

        try:
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.canonical_to_aliases = data.get("canonical_to_aliases", {})
            self.alias_to_canonical = data.get("alias_to_canonical", {})

        except Exception as e:
            # If loading fails, start with empty lexicon
            print(f"Warning: Failed to load lexicon from {self.lexicon_path}: {e}")
            self.canonical_to_aliases = {}
            self.alias_to_canonical = {}

    def save(self) -> None:
        """Save lexicon to disk."""
        # Ensure output directory exists
        lexicon_dir = os.path.dirname(self.lexicon_path)
        if lexicon_dir:  # Only create directory if path has a parent directory
            os.makedirs(lexicon_dir, exist_ok=True)

        data = {
            "canonical_to_aliases": self.canonical_to_aliases,
            "alias_to_canonical": self.alias_to_canonical
        }

        with open(self.lexicon_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def normalize_entity(self, raw_entity: str) -> str:
        """Normalize a raw entity using the lexicon.

        Strategy:
        1. Apply string normalization (trim, lowercase, etc.)
        2. Check if normalized entity exists in alias_to_canonical
           - If hit: return canonical
           - If miss: create new canonical and update lexicon

        Args:
            raw_entity: Raw entity string from LLM extraction

        Returns:
            Canonical entity string
        """
        # Step 1: String normalization
        norm_entity = normalize_string(raw_entity)

        if not norm_entity:
            return ""

        # Step 2: Lookup in lexicon
        if norm_entity in self.alias_to_canonical:
            # Hit: return existing canonical
            return self.alias_to_canonical[norm_entity]

        # Miss: create new canonical
        canonical = norm_entity

        # Add to lexicon
        self._add_canonical(canonical, norm_entity)

        return canonical

    def _add_canonical(self, canonical: str, alias: str) -> None:
        """Add a new canonical entity with an alias to the lexicon.

        Args:
            canonical: Canonical entity string
            alias: Alias string (may be same as canonical)
        """
        # Add to canonical_to_aliases
        if canonical not in self.canonical_to_aliases:
            self.canonical_to_aliases[canonical] = []

        if alias not in self.canonical_to_aliases[canonical]:
            self.canonical_to_aliases[canonical].append(alias)

        # Add to alias_to_canonical
        self.alias_to_canonical[alias] = canonical

    def normalize_entities(self, raw_entities: List[str]) -> List[str]:
        """Normalize a list of raw entities.

        Args:
            raw_entities: List of raw entity strings

        Returns:
            List of canonical entities (deduplicated)
        """
        canonical_set: Set[str] = set()

        for raw_entity in raw_entities:
            canonical = self.normalize_entity(raw_entity)
            if canonical:  # Skip empty strings
                canonical_set.add(canonical)

        return list(canonical_set)

    def get_stats(self) -> Dict[str, int]:
        """Get lexicon statistics.

        Returns:
            Dict with 'num_canonicals' and 'num_aliases'
        """
        return {
            "num_canonicals": len(self.canonical_to_aliases),
            "num_aliases": len(self.alias_to_canonical)
        }
