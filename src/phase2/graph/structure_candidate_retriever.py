"""Step 5b: Structure-based Candidate Retrieval and S_graph Calculation.

Single-edge (1-edge) entity-path similarity:
- Enumerate only 1-edge path tokens
- Compute one S_graph over 1-edge token sets

For entity-aware relaxed mode, two paths are considered overlapping if:
- their role sequence is the same, and
- entity overlap count >= ceil(n/2), where n is path entity count.
"""
import json
import os
import math
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from src.shared.logger import logger


class StructureCandidateRetriever:
    """Retrieves candidate experiences based on multi-length entity-connected path similarity."""

    def __init__(
        self,
        top_k: int = 40,
        lexicon_path: str = "data/lexicon/lexicon.json",
        path_token_mode: str = "hybrid"
    ):
        """Initialize structure candidate retriever.

        Args:
            top_k: Number of top candidates to retain per experience
            lexicon_path: Path to alias-to-canonical mapping
            path_token_mode: Path tokenization mode for S_k comparison:
                - "hybrid": S1 uses Role:Entity token, S2/3/4 use role-only tokens
                - "role_only": Legacy mode, all S1~S4 use role-only tokens
                - "entity_aware": All S1~S4 use Role:Entity tokens
                - "entity_aware_relaxed": All S1~S4 use Role:Entity tokens;
                  path overlap is relaxed by entity overlap threshold:
                  overlap_entities >= ceil(n/2)
                - "entity_aware_relaxed_s1_role_only": S1 uses role-only exact match;
                  S2/3/4 use Role:Entity with relaxed overlap rule.
        """
        self.top_k = top_k
        valid_modes = {
            "hybrid",
            "role_only",
            "entity_aware",
            "entity_aware_relaxed",
            "entity_aware_relaxed_s1_role_only"
        }
        if path_token_mode not in valid_modes:
            raise ValueError(f"Invalid path_token_mode: {path_token_mode}. Must be one of {sorted(valid_modes)}")
        self.path_token_mode = path_token_mode

        self.alias_to_canonical = self._load_alias_mapping(lexicon_path)

    def _load_alias_mapping(self, lexicon_path: str) -> Dict[str, str]:
        """Load alias->canonical mapping used for entity-edge canonicalization."""
        if not os.path.exists(lexicon_path):
            logger.log_warning(f"Lexicon not found: {lexicon_path}. Step 5b will use raw entity_edges.")
            return {}

        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
            mapping = lexicon.get('alias_to_canonical', {})
            logger.log_info(f"Loaded lexicon aliases: {len(mapping)} from {lexicon_path}")
            return mapping
        except Exception as e:
            logger.log_warning(f"Failed to load lexicon from {lexicon_path}: {e}")
            return {}

    def _canonicalize_entity(self, entity: str) -> str:
        """Map entity mention to canonical form for Step 5b path construction."""
        if not entity:
            return ""

        # Prefer exact mapping from lexicon
        if entity in self.alias_to_canonical:
            return self.alias_to_canonical[entity]

        # Fallback: lower + trim + collapse spaces
        normalized = " ".join(entity.strip().lower().split())
        return self.alias_to_canonical.get(normalized, normalized)

    def load_entity_candidates(self, entity_candidates_path: str) -> Dict[str, List[str]]:
        """Load entity candidates from JSONL file.

        Args:
            entity_candidates_path: Path to entity_candidates.jsonl

        Returns:
            Dict mapping exp_id to list of candidate_ids
        """
        entity_candidates = {}

        with open(entity_candidates_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                exp_id = item['exp_id']
                candidate_ids = [c['candidate_id'] for c in item['candidates']]
                entity_candidates[exp_id] = candidate_ids

        return entity_candidates

    def _parse_roles(self, edge_type: str) -> Tuple[str, str]:
        """Parse from/to roles from edge type like 'Condition→Action'."""
        if not edge_type:
            return "Unknown", "Unknown"
        if "→" in edge_type:
            parts = edge_type.split("→", 1)
            return parts[0].strip() or "Unknown", parts[1].strip() or "Unknown"
        if "->" in edge_type:
            parts = edge_type.split("->", 1)
            return parts[0].strip() or "Unknown", parts[1].strip() or "Unknown"
        return "Unknown", "Unknown"

    def build_entity_graph(self, entity_edges: List[Dict]) -> Dict[str, List[Tuple[str, str, str]]]:
        """Build directed graph from entity edges with canonicalized entities.

        Args:
            entity_edges: List of entity edge dicts with 'edge', 'from_entity', 'to_entity'

        Returns:
            Adjacency list:
            {from_node: [(to_node, role_edge_token, entity_edge_token), ...]}
            where:
            - role_edge_token: "Condition→Action"
            - entity_edge_token: "Condition:sinusitis→Action:iv antibiotics"
        """
        graph = defaultdict(list)

        for e_edge in entity_edges:
            edge_type = e_edge.get('edge', '')
            from_entity = self._canonicalize_entity(e_edge.get('from_entity', ''))
            to_entity = self._canonicalize_entity(e_edge.get('to_entity', ''))

            if edge_type and from_entity and to_entity:
                from_role, to_role = self._parse_roles(edge_type)
                entity_edge_token = f"{from_role}:{from_entity}→{to_role}:{to_entity}"

                # Connectivity node choice:
                # - entity_aware: node is Role:Entity (strictest)
                # - hybrid / role_only: node is canonical entity (legacy-compatible)
                if self.path_token_mode in {"entity_aware", "entity_aware_relaxed"}:
                    from_node = f"{from_role}:{from_entity}"
                    to_node = f"{to_role}:{to_entity}"
                else:
                    from_node = from_entity
                    to_node = to_entity

                graph[from_node].append((to_node, edge_type, entity_edge_token))

        return dict(graph)

    def enumerate_entity_paths(
        self,
        graph: Dict[str, List[Tuple[str, str, str]]],
        max_length: int = 4
    ) -> Dict[int, Set[str]]:
        """Enumerate entity-connected paths of length 1-4.

        Paths are enumerated based on entity connectivity:
        edge1.to_entity == edge2.from_entity

        Args:
            graph: Entity graph {from_entity: [(to_entity, edge_type, to_entity), ...]}
            max_length: Maximum path length (number of edges)

        Returns:
            Dict mapping path_length to set of path tokens.
            - entity-aware mode token example:
              {1: {'Condition:sinusitis→Action:iv antibiotics'},
               2: {'Condition:sinusitis→Action:iv antibiotics→Outcome:prevent complications'}}
            - legacy mode token example:
              {1: {'Condition→Action'}, 2: {'Condition→Action|Action→Outcome'}}
        """
        all_paths = {1: set(), 2: set(), 3: set(), 4: set()}

        # Helper function for DFS path enumeration
        def dfs_paths(
            current_entity: str,
            role_edge_seq: List[str],
            entity_edge_seq: List[str],
            visited_entities: Set[str]
        ):
            """Enumerate all entity-connected paths starting from current entity."""
            path_length = len(role_edge_seq)

            # Record current path (if length >= 1)
            if path_length >= 1 and path_length <= max_length:
                if self.path_token_mode in {"entity_aware", "entity_aware_relaxed"}:
                    # edge_sequence entries are 1-edge tokens like "Role1:ent1→Role2:ent2"
                    # Build node chain token:
                    # e1: A→B, e2: B→C => A→B→C
                    nodes = []
                    for i, edge_token in enumerate(entity_edge_seq):
                        left, right = edge_token.split('→', 1)
                        if i == 0:
                            nodes.append(left)
                        nodes.append(right)
                    path_str = '→'.join(nodes)
                elif self.path_token_mode == "hybrid":
                    # Hybrid:
                    # - S1 uses Role:Entity edge token
                    # - S2/S3/S4 use role-only edge sequences
                    if path_length == 1:
                        path_str = entity_edge_seq[0]
                    else:
                        path_str = '|'.join(role_edge_seq)
                elif self.path_token_mode == "entity_aware_relaxed_s1_role_only":
                    # S1 role-only; S2/3/4 entity-aware tokens
                    if path_length == 1:
                        path_str = role_edge_seq[0]
                    else:
                        nodes = []
                        for i, edge_token in enumerate(entity_edge_seq):
                            left, right = edge_token.split('→', 1)
                            if i == 0:
                                nodes.append(left)
                            nodes.append(right)
                        path_str = '→'.join(nodes)
                else:
                    # Legacy role-only token
                    path_str = '|'.join(role_edge_seq)
                all_paths[path_length].add(path_str)

            # Stop if reached max length
            if path_length >= max_length:
                return

            # Continue exploring from current_entity
            if current_entity in graph:
                for next_entity, role_edge_token, entity_edge_token in graph[current_entity]:
                    # Only follow if next_entity not visited (avoid cycles)
                    if next_entity not in visited_entities:
                        dfs_paths(
                            next_entity,
                            role_edge_seq + [role_edge_token],
                            entity_edge_seq + [entity_edge_token],
                            visited_entities | {next_entity}
                        )

        # Start DFS from each entity
        all_entities = set(graph.keys())
        for target_list in graph.values():
            for to_entity, _, _ in target_list:
                all_entities.add(to_entity)

        for start_entity in all_entities:
            dfs_paths(start_entity, [], [], {start_entity})

        return all_paths

    def compute_jaccard_similarity(
        self,
        set1: Set[str],
        set2: Set[str]
    ) -> float:
        """Compute Jaccard similarity between two sets.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Jaccard similarity: |A ∩ B| / |A ∪ B|
        """
        if not set1 and not set2:
            return 0.0

        if self.path_token_mode == "entity_aware_relaxed":
            intersection_size = self._relaxed_intersection_size(set1, set2)
            union_size = len(set1) + len(set2) - intersection_size
            return intersection_size / union_size if union_size > 0 else 0.0

        union = set1 | set2
        if not union:
            return 0.0

        intersection = set1 & set2
        return len(intersection) / len(union)

    def _exact_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute exact-token Jaccard similarity."""
        if not set1 and not set2:
            return 0.0
        union = set1 | set2
        if not union:
            return 0.0
        return len(set1 & set2) / len(union)

    def _parse_entity_aware_path(self, path_token: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Parse entity-aware path token into (role_seq, entity_seq)."""
        roles = []
        entities = []
        for node in path_token.split('→'):
            role, sep, entity = node.partition(':')
            if not sep:
                # Fallback for malformed token
                roles.append("Unknown")
                entities.append(node.strip().lower())
            else:
                roles.append(role.strip())
                entities.append(entity.strip().lower())
        return tuple(roles), tuple(entities)

    def _entities_overlap_enough(self, entities1: Tuple[str, ...], entities2: Tuple[str, ...]) -> bool:
        """Relaxed overlap rule: overlap >= ceil(n/2), where n is path entity count."""
        n = len(entities1)
        threshold = math.ceil(n / 2)
        overlap = len(set(entities1) & set(entities2))
        return overlap >= threshold

    def _max_bipartite_matching_size(self, left_adj: List[List[int]], right_size: int) -> int:
        """Compute maximum bipartite matching cardinality (Kuhn algorithm)."""
        match_r = [-1] * right_size

        def try_augment(u: int, seen: List[bool]) -> bool:
            for v in left_adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                if match_r[v] == -1 or try_augment(match_r[v], seen):
                    match_r[v] = u
                    return True
            return False

        matched = 0
        for u in range(len(left_adj)):
            seen = [False] * right_size
            if try_augment(u, seen):
                matched += 1
        return matched

    def _relaxed_intersection_size(self, set1: Set[str], set2: Set[str]) -> int:
        """Intersection size under relaxed entity-overlap rule for entity-aware paths."""
        # Group parsed paths by role sequence; only same role sequence can match
        grouped1 = defaultdict(list)
        grouped2 = defaultdict(list)

        for token in set1:
            role_seq, entity_seq = self._parse_entity_aware_path(token)
            grouped1[role_seq].append(entity_seq)
        for token in set2:
            role_seq, entity_seq = self._parse_entity_aware_path(token)
            grouped2[role_seq].append(entity_seq)

        intersection_size = 0
        for role_seq in set(grouped1.keys()) & set(grouped2.keys()):
            left = grouped1[role_seq]
            right = grouped2[role_seq]

            # Build bipartite graph edges by relaxed overlap rule
            adj = []
            for e1 in left:
                neighbors = []
                for j, e2 in enumerate(right):
                    if self._entities_overlap_enough(e1, e2):
                        neighbors.append(j)
                adj.append(neighbors)

            intersection_size += self._max_bipartite_matching_size(adj, len(right))

        return intersection_size

    def compute_pooled_similarity(
        self,
        paths1: Dict[int, Set[str]],
        paths2: Dict[int, Set[str]]
    ) -> float:
        """Compute S_graph from 1-edge path token sets only."""
        pooled1 = set(paths1.get(1, set()))
        pooled2 = set(paths2.get(1, set()))

        # Enforce relaxed role-entity matching for pooled similarity
        intersection_size = self._relaxed_intersection_size(pooled1, pooled2)
        union_size = len(pooled1) + len(pooled2) - intersection_size
        return intersection_size / union_size if union_size > 0 else 0.0

    def retrieve_candidates(
        self,
        experiences: List[Dict[str, Any]],
        entity_candidates_path: str
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K structure candidates and compute S_graph.

        Only computes S_graph for candidate pairs from entity_candidates.

        Args:
            experiences: List of experience dicts with 'id' and 'entity_edges' fields
            entity_candidates_path: Path to entity_candidates.jsonl

        Returns:
            List of candidate dicts, one per experience:
            {
                "exp_id": str,
                "candidates": [
                    {
                        "candidate_id": str,
                        "s_graph": float
                    },
                    ...
                ]
            }
        """
        logger.log_info(f"\n{'='*80}")
        logger.log_info(f"Step 5b: Structure Candidate Retrieval + S_graph (Entity-Connected Paths)")
        logger.log_info(f"{'='*80}")
        logger.log_info(f"Total experiences: {len(experiences)}")
        logger.log_info(f"Top-K candidates per experience: {self.top_k}")
        mode_desc = {
            "hybrid": "hybrid (S1 role+entity; S2/3/4 role-only)",
            "role_only": "role-only (legacy)",
            "entity_aware": "entity-aware strict (Role:Entity exact match for all S1~S4)",
            "entity_aware_relaxed": "entity-aware relaxed (Role:Entity; overlap>=ceil(n/2))",
            "entity_aware_relaxed_s1_role_only": "S1 role-only; S2/3/4 entity-aware relaxed"
        }
        logger.log_info(f"Path token mode: {mode_desc[self.path_token_mode]}")
        logger.log_info("Similarity mode: 1-edge only (single S_graph)")

        # Load entity candidates
        logger.log_info(f"\nLoading entity candidates from: {entity_candidates_path}")
        entity_candidates = self.load_entity_candidates(entity_candidates_path)
        logger.log_info(f"Loaded entity candidates for {len(entity_candidates)} experiences")

        # Build experience ID to index mapping
        exp_id_to_idx = {exp['id']: idx for idx, exp in enumerate(experiences)}

        # Pre-compute entity-connected paths for all experiences
        logger.log_info("Enumerating entity-connected paths for all experiences...")
        exp_paths = []

        for exp in experiences:
            entity_edges = exp.get('entity_edges', [])

            # Build entity graph
            graph = self.build_entity_graph(entity_edges)

            # Enumerate only 1-edge paths
            paths = self.enumerate_entity_paths(graph, max_length=1)

            exp_paths.append(paths)

        logger.log_info(f"Path enumeration complete")

        # Compute S_graph for entity candidate pairs
        logger.log_info("Computing S_graph for entity candidate pairs...")
        results = []

        for idx, exp in enumerate(experiences):
            if (idx + 1) % 100 == 0:
                logger.log_info(f"  Processing {idx + 1}/{len(experiences)}")

            exp_id = exp['id']
            exp_path_sets = exp_paths[idx]

            # Get entity candidates for this experience
            candidate_ids = entity_candidates.get(exp_id, [])

            # Compute S_graph for each candidate
            candidate_scores = []
            for cand_id in candidate_ids:
                if cand_id not in exp_id_to_idx:
                    continue

                cand_idx = exp_id_to_idx[cand_id]
                cand_path_sets = exp_paths[cand_idx]

                # Compute pooled similarity
                s_graph = self.compute_pooled_similarity(
                    exp_path_sets,
                    cand_path_sets
                )

                candidate_scores.append({
                    'candidate_id': cand_id,
                    's_graph': s_graph
                })

            # Sort by S_graph and retain top-K
            candidate_scores.sort(key=lambda x: x['s_graph'], reverse=True)
            top_candidates = candidate_scores[:self.top_k]

            results.append({
                'exp_id': exp_id,
                'candidates': top_candidates
            })

        # Summary statistics
        total_candidates = sum(len(r['candidates']) for r in results)
        avg_candidates = total_candidates / len(results) if results else 0

        logger.log_info(f"\nStructure Candidate Retrieval Summary:")
        logger.log_info(f"  Total candidate pairs: {total_candidates}")
        logger.log_info(f"  Average candidates per experience: {avg_candidates:.2f}") # Count experiences with < top_k candidates
        low_recall = sum(1 for r in results if len(r['candidates']) < self.top_k)
        logger.log_info(f"  Experiences with < {self.top_k} candidates: {low_recall} ({low_recall/len(results)*100:.1f}%)")

        # S_graph distribution
        all_scores = [c['s_graph'] for r in results for c in r['candidates']]
        if all_scores:
            import numpy as np
            scores_array = np.array(all_scores)

            logger.log_info(f"\n  S_graph statistics:")
            logger.log_info(f"    Min: {np.min(scores_array):.4f}")
            logger.log_info(f"    Max: {np.max(scores_array):.4f}")
            logger.log_info(f"    Mean: {np.mean(scores_array):.4f}")
            logger.log_info(f"    Median: {np.median(scores_array):.4f}")
            logger.log_info(f"    Std: {np.std(scores_array):.4f}")

            # Distribution bins
            logger.log_info(f"\n  S_graph distribution:")
            bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            for i, (low, high) in enumerate(bins):
                if i == len(bins) - 1:
                    # Last bin includes upper bound
                    count = np.sum((scores_array >= low) & (scores_array <= high))
                    logger.log_info(f"    [{low:.1f}, {high:.1f}]: {count:5d} ({pct if (pct := count / len(scores_array) * 100) else 0:.1f}%)")
                else:
                    count = np.sum((scores_array >= low) & (scores_array < high))
                    logger.log_info(f"    [{low:.1f}, {high:.1f}): {count:5d} ({pct if (pct := count / len(scores_array) * 100) else 0:.1f}%)")

        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save structure candidate results to JSONL file.

        Args:
            results: List of candidate retrieval results
            output_path: Path to save JSONL file
        """
        logger.log_info(f"\nSaving structure candidates to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        logger.log_info(f"Saved {len(results)} structure candidate lists")
