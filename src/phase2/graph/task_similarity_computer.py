"""Step 5d: Task Similarity Computation."""
import json
from typing import List, Dict, Any

from src.shared.logger import logger


class TaskSimilarityComputer:
    """Computes task similarity based on task_type_norm field."""

    def compute_task_similarity(
        self,
        semantic_results: List[Dict[str, Any]],
        experiences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute task similarity for candidate pairs.

        Task similarity is binary:
        - s_task = 1.0 if both experiences have the same task_type_norm
        - s_task = 0.0 otherwise

        Args:
            semantic_results: Results from Step 5c. Supports:
                - Undirected format: {exp_a, exp_b, s_ent, s_graph, s_sem, ...}
                - Directed format: {exp_id, candidates:[...]}
            experiences: List of all experiences

        Returns:
            Directed candidate-list format:
            {
              "exp_id": str,
              "candidates": [{"candidate_id", "s_ent", "s_graph", "s_sem", "s_task"}]
            }
        """
        logger.log_info(f"\n{'='*80}")
        logger.log_info(f"Step 5d: Task Similarity Computation")
        logger.log_info(f"{'='*80}")

        # Build exp_id to experience mapping
        exp_lookup = {exp['id']: exp for exp in experiences}

        # Always produce directed output lists for all experiences
        result_map = {exp['id']: [] for exp in experiences}
        total_pairs = 0
        same_task_count = 0

        # Undirected format from Step 5c: append to both directions
        if semantic_results and 'exp_a' in semantic_results[0] and 'exp_b' in semantic_results[0]:
            total_pairs_target = len(semantic_results)
            for item in semantic_results:
                exp_a = item['exp_a']
                exp_b = item['exp_b']
                exp1 = exp_lookup.get(exp_a)
                exp2 = exp_lookup.get(exp_b)

                if not exp1:
                    logger.log_warning(f"Experience {exp_a} not found, skipping")
                    continue
                if not exp2:
                    logger.log_warning(f"Experience {exp_b} not found, skipping")
                    continue

                task_a = exp1.get('task_type_norm', 'unknown')
                task_b = exp2.get('task_type_norm', 'unknown')
                s_task = 1.0 if task_a == task_b else 0.0

                if s_task == 1.0:
                    same_task_count += 1
                total_pairs += 1
                logger.log_info(f"  Processed {total_pairs}/{total_pairs_target} pairs")

                if exp_a in result_map:
                    result_map[exp_a].append({
                        'candidate_id': exp_b,
                        's_ent': item['s_ent'],
                        's_graph': item['s_graph'],
                        's_sem': item['s_sem'],
                        's_task': s_task
                    })
                if exp_b in result_map:
                    result_map[exp_b].append({
                        'candidate_id': exp_a,
                        's_ent': item['s_ent'],
                        's_graph': item['s_graph'],
                        's_sem': item['s_sem'],
                        's_task': s_task
                    })

        # Directed format
        else:
            total_pairs_target = sum(len(item.get('candidates', [])) for item in semantic_results)
            for item in semantic_results:
                exp_id = item['exp_id']
                exp = exp_lookup.get(exp_id)

                if not exp:
                    logger.log_warning(f"Experience {exp_id} not found, skipping")
                    continue

                exp_task = exp.get('task_type_norm', 'unknown')

                updated_candidates = []

                for cand in item['candidates']:
                    cand_id = cand['candidate_id']
                    cand_exp = exp_lookup.get(cand_id)

                    if not cand_exp:
                        logger.log_warning(f"Candidate {cand_id} not found, skipping")
                        continue

                    cand_task = cand_exp.get('task_type_norm', 'unknown')
                    s_task = 1.0 if exp_task == cand_task else 0.0

                    if s_task == 1.0:
                        same_task_count += 1

                    updated_candidates.append({
                        'candidate_id': cand_id,
                        's_ent': cand['s_ent'],
                        's_graph': cand['s_graph'],
                        's_sem': cand['s_sem'],
                        's_task': s_task
                    })

                    total_pairs += 1
                    logger.log_info(f"  Processed {total_pairs}/{total_pairs_target} pairs")

                if exp_id in result_map:
                    result_map[exp_id].extend(updated_candidates)

        # Summary statistics
        same_task_ratio = same_task_count / total_pairs if total_pairs > 0 else 0

        logger.log_info(f"\nTask Similarity Computation Summary:")
        logger.log_info(f"  Total pairs evaluated: {total_pairs}")
        logger.log_info(f"  Same task pairs: {same_task_count} ({same_task_ratio*100:.1f}%)")
        logger.log_info(f"  Different task pairs: {total_pairs - same_task_count} ({(1-same_task_ratio)*100:.1f}%)")

        # Stable output order: follow experiences order
        results = []
        for exp in experiences:
            exp_id = exp['id']
            results.append({
                'exp_id': exp_id,
                'candidates': result_map.get(exp_id, [])
            })

        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save task similarity results to JSONL file.

        Args:
            results: List of task similarity results
            output_path: Path to save JSONL file
        """
        logger.log_info(f"\nSaving task scores to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        logger.log_info(f"Saved {len(results)} task score lists")
