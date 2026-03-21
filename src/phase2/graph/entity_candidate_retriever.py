"""Step 5a: Entity-based Candidate Retrieval and S_ent Calculation."""
import json
import math
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.shared.logger import logger


class EntityCandidateRetriever:
    """Retrieves candidate experiences based on entity overlap and computes S_ent."""

    def __init__(
        self,
        entity_stats_path: str,
        entity_postings_path: str,
        top_k: int = 60
    ):
        """Initialize entity candidate retriever.

        Args:
            entity_stats_path: Path to entity_stats.json (contains IDF values)
            entity_postings_path: Path to entity_postings.json (inverted index)
            top_k: Number of top candidates to retain per experience
        """
        self.top_k = top_k

        # Load entity stats (IDF)
        logger.log_info(f"Loading entity stats from: {entity_stats_path}")
        with open(entity_stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            self.N = stats['N']
            self.idf = stats.get('idf', {})

        # Load entity postings (inverted index)
        logger.log_info(f"Loading entity postings from: {entity_postings_path}")
        with open(entity_postings_path, 'r', encoding='utf-8') as f:
            self.entity_postings = json.load(f)

        logger.log_info(f"Loaded {len(self.idf)} entities with IDF scores")
        logger.log_info(f"Loaded {len(self.entity_postings)} entity postings")

    def compute_tfidf_vector(self, entities: List[str]) -> Dict[str, float]:
        """Compute TF-IDF vector for a list of entities.

        Args:
            entities: List of entity strings

        Returns:
            Dict mapping entity to TF-IDF score
        """
        # Compute term frequency (TF)
        tf = defaultdict(int)
        for entity in entities:
            tf[entity] += 1

        # Normalize by total count
        total = len(entities)
        if total == 0:
            return {}

        # Compute TF-IDF
        tfidf = {}
        for entity, count in tf.items():
            tf_score = count / total
            idf_score = self.idf.get(entity, 0.0)
            tfidf[entity] = tf_score * idf_score

        return tfidf

    def cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> float:
        """Compute cosine similarity between two TF-IDF vectors.

        Args:
            vec1: TF-IDF vector (entity -> score)
            vec2: TF-IDF vector (entity -> score)

        Returns:
            Cosine similarity score (0-1)
        """
        # Compute dot product
        common_entities = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[e] * vec2[e] for e in common_entities)

        # Compute magnitudes
        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def retrieve_candidates(
        self,
        experiences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K entity candidates for each experience and compute S_ent.

        Args:
            experiences: List of experience dicts with 'id' and 'canonical_entities' fields

        Returns:
            List of candidate dicts, one per experience:
            {
                "exp_id": str,
                "candidates": [
                    {
                        "candidate_id": str,
                        "s_ent": float,
                        "shared_entities": int
                    },
                    ...
                ],
                "tfidf_vector": Dict[str, float]  # for later use
            }
        """
        logger.log_info(f"\n{'='*80}")
        logger.log_info(f"Step 5a: Entity Candidate Retrieval + S_ent Computation")
        logger.log_info(f"{'='*80}")
        logger.log_info(f"Total experiences: {len(experiences)}")
        logger.log_info(f"Top-K candidates per experience: {self.top_k}")

        # Build experience ID to index mapping
        exp_id_to_idx = {exp['id']: idx for idx, exp in enumerate(experiences)}

        # Pre-compute TF-IDF vectors for all experiences
        logger.log_info("Computing TF-IDF vectors for all experiences...")
        tfidf_vectors = []
        for exp in experiences:
            tfidf = self.compute_tfidf_vector(exp.get('canonical_entities', []))
            tfidf_vectors.append(tfidf)

        logger.log_info("Retrieving candidates and computing S_ent...")
        results = []

        for idx, exp in enumerate(experiences):
            if (idx + 1) % 100 == 0:
                logger.log_info(f"  Processing {idx + 1}/{len(experiences)}")

            exp_id = exp['id']
            entities = exp.get('canonical_entities', [])
            tfidf_vec = tfidf_vectors[idx]

            # Candidate retrieval: find all experiences sharing at least one entity
            candidate_ids = set()
            for entity in entities:
                if entity in self.entity_postings:
                    candidate_ids.update(self.entity_postings[entity])

            # Remove self
            candidate_ids.discard(exp_id)

            # Compute S_ent for each candidate
            candidate_scores = []
            for cand_id in candidate_ids:
                if cand_id not in exp_id_to_idx:
                    continue

                cand_idx = exp_id_to_idx[cand_id]
                cand_tfidf = tfidf_vectors[cand_idx]

                # Compute S_ent (cosine similarity of TF-IDF vectors)
                s_ent = self.cosine_similarity(tfidf_vec, cand_tfidf)

                # Count shared entities
                shared_entities = len(set(entities) & set(experiences[cand_idx]['canonical_entities']))

                candidate_scores.append({
                    'candidate_id': cand_id,
                    's_ent': s_ent,
                    'shared_entities': shared_entities
                })

            # Sort by S_ent and retain top-K
            candidate_scores.sort(key=lambda x: x['s_ent'], reverse=True)
            top_candidates = candidate_scores[:self.top_k]

            results.append({
                'exp_id': exp_id,
                'candidates': top_candidates,
                'tfidf_vector': tfidf_vec
            })

        # Summary statistics
        total_candidates = sum(len(r['candidates']) for r in results)
        avg_candidates = total_candidates / len(results) if results else 0

        logger.log_info(f"\nEntity Candidate Retrieval Summary:")
        logger.log_info(f"  Total candidate pairs: {total_candidates}")
        logger.log_info(f"  Average candidates per experience: {avg_candidates:.2f}")

        # Count experiences with < top_k candidates
        low_recall = sum(1 for r in results if len(r['candidates']) < self.top_k)
        logger.log_info(f"  Experiences with < {self.top_k} candidates: {low_recall} ({low_recall/len(results)*100:.1f}%)")

        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save entity candidate results to JSONL file.

        Args:
            results: List of candidate retrieval results
            output_path: Path to save JSONL file
        """
        logger.log_info(f"\nSaving entity candidates to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                # Remove tfidf_vector from output (too large)
                output_item = {
                    'exp_id': result['exp_id'],
                    'candidates': result['candidates']
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

        logger.log_info(f"Saved {len(results)} entity candidate lists")
