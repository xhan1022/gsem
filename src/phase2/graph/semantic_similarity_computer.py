"""Step 5c: Merge Candidates and Compute Semantic Similarity.

This step computes semantic similarity on undirected unique pairs to avoid
duplicate LLM calls for (A, B) and (B, A).
"""
import json
import re
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.shared.logger import logger
from src.phase2.graph.prompts import (
    SEMANTIC_SIMILARITY_SYSTEM_PROMPT,
    SEMANTIC_SIMILARITY_HUMAN_PROMPT
)


class SemanticSimilarityComputer:
    """Merges candidates, deduplicates into undirected pairs, then computes S_sem."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        api_key: str = None,
        merged_top_k: int = 100,
        temperature: float = 0.0
    ):
        """Initialize semantic similarity computer.

        Args:
            model_name: LLM model name
            api_base: API base URL
            api_key: API key
            merged_top_k: Maximum candidates after merging
            temperature: LLM temperature
        """
        self.merged_top_k = merged_top_k

        # Initialize LLM
        logger.log_info(f"Initializing LLM: {model_name}")
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
            temperature=temperature,
            request_timeout=120
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SEMANTIC_SIMILARITY_SYSTEM_PROMPT),
            ("human", SEMANTIC_SIMILARITY_HUMAN_PROMPT)
        ])

        self.chain = self.prompt | self.llm

    def merge_candidates(
        self,
        entity_candidates: List[Dict[str, Any]],
        structure_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge entity and structure candidates and retain top-K.

        Args:
            entity_candidates: List of entity candidate results (from Step 5a)
            structure_candidates: List of structure candidate results (from Step 5b)

        Returns:
            List of merged candidate results:
            {
                "exp_id": str,
                "candidates": [
                    {
                        "candidate_id": str,
                        "s_ent": float,
                        "s_graph": float,
                        "combined_score": float  # s_ent + s_graph
                    },
                    ...
                ]
            }
        """
        logger.log_info(f"\n{'='*80}")
        logger.log_info(f"Step 5c: Merge Candidates")
        logger.log_info(f"{'='*80}")

        # Build lookup dicts
        entity_lookup = {item['exp_id']: item['candidates'] for item in entity_candidates}
        structure_lookup = {item['exp_id']: item['candidates'] for item in structure_candidates}

        merged_results = []

        for exp_id in entity_lookup.keys():
            ent_cands = entity_lookup.get(exp_id, [])
            struct_cands = structure_lookup.get(exp_id, [])

            # Build candidate dict: candidate_id -> scores
            candidate_dict = {}

            # Add entity candidates
            for cand in ent_cands:
                cand_id = cand['candidate_id']
                candidate_dict[cand_id] = {
                    's_ent': cand['s_ent'],
                    's_graph': 0.0
                }

            # Add structure candidates (merge if exists)
            for cand in struct_cands:
                cand_id = cand['candidate_id']
                if cand_id in candidate_dict:
                    candidate_dict[cand_id]['s_graph'] = cand['s_graph']
                else:
                    candidate_dict[cand_id] = {
                        's_ent': 0.0,
                        's_graph': cand['s_graph']
                    }

            # Compute combined score and convert to list
            merged_cands = []
            for cand_id, scores in candidate_dict.items():
                merged_cands.append({
                    'candidate_id': cand_id,
                    's_ent': scores['s_ent'],
                    's_graph': scores['s_graph'],
                    'combined_score': scores['s_ent'] + scores['s_graph']
                })

            # Sort by combined score and retain top-K
            merged_cands.sort(key=lambda x: x['combined_score'], reverse=True)
            top_cands = merged_cands[:self.merged_top_k]

            merged_results.append({
                'exp_id': exp_id,
                'candidates': top_cands
            })

        # Summary statistics
        total_candidates = sum(len(r['candidates']) for r in merged_results)
        avg_candidates = total_candidates / len(merged_results) if merged_results else 0

        logger.log_info(f"Merged Candidate Summary:")
        logger.log_info(f"  Total candidate pairs: {total_candidates}")
        logger.log_info(f"  Average candidates per experience: {avg_candidates:.2f}")

        return merged_results

    def build_undirected_pairs(
        self,
        merged_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert directed candidate lists to undirected unique pairs.

        If the same pair appears in both directions, retain one pair and merge
        scores using max to preserve the stronger signal after top-k truncation.
        """
        pair_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for item in merged_candidates:
            src_id = item['exp_id']
            for cand in item.get('candidates', []):
                dst_id = cand['candidate_id']
                if src_id == dst_id:
                    continue

                exp_a, exp_b = sorted((src_id, dst_id))
                key = (exp_a, exp_b)

                if key not in pair_map:
                    pair_map[key] = {
                        'exp_a': exp_a,
                        'exp_b': exp_b,
                        's_ent': float(cand.get('s_ent', 0.0)),
                        's_graph': float(cand.get('s_graph', 0.0))
                    }
                else:
                    pair_map[key]['s_ent'] = max(pair_map[key]['s_ent'], float(cand.get('s_ent', 0.0)))
                    pair_map[key]['s_graph'] = max(pair_map[key]['s_graph'], float(cand.get('s_graph', 0.0)))

        undirected_pairs = list(pair_map.values())
        undirected_pairs.sort(key=lambda x: x['s_ent'] + x['s_graph'], reverse=True)

        logger.log_info(f"Undirected Pair Summary:")
        logger.log_info(f"  Unique undirected pairs: {len(undirected_pairs)}")

        return undirected_pairs

    def compute_semantic_similarity(
        self,
        merged_candidates: List[Dict[str, Any]],
        experiences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute semantic similarity with undirected evaluation + directed output.

        Args:
            merged_candidates: Directed merged candidate lists
            experiences: List of all experiences

        Returns:
            Directed candidate-list format (aligned with entity_candidates):
            {
                "exp_id": str,
                "candidates": [
                    {
                        "candidate_id": str,
                        "s_ent": float,
                        "s_graph": float,
                        "s_sem": float,
                        "s_sem_reason": str
                    },
                    ...
                ]
            }
        """
        logger.log_info(f"\n{'='*80}")
        logger.log_info(f"Step 5c: Semantic Similarity Computation (LLM, Undirected)")
        logger.log_info(f"{'='*80}")

        # Build exp_id to experience mapping
        exp_lookup = {exp['id']: exp for exp in experiences}

        # Build undirected unique pairs first
        undirected_pairs = self.build_undirected_pairs(merged_candidates)
        total_pairs = len(undirected_pairs)
        logger.log_info(f"Total candidate pairs to evaluate: {total_pairs}")

        undirected_results = []
        processed_pairs = 0

        for item in undirected_pairs:
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

            # Call LLM once per undirected pair
            s_sem, reason = self._compute_pair_semantic_similarity(exp1, exp2)

            undirected_results.append({
                'exp_a': exp_a,
                'exp_b': exp_b,
                's_ent': item['s_ent'],
                's_graph': item['s_graph'],
                's_sem': s_sem,
                's_sem_reason': reason
            })

            processed_pairs += 1
            logger.log_info(f"  Processed {processed_pairs}/{total_pairs} pairs")

        logger.log_info(f"\nSemantic Similarity Computation Complete:")
        logger.log_info(f"  Total pairs evaluated: {processed_pairs}")

        # Convert back to directed candidate-list format to align with Step 5a/outputs.
        score_lookup = {}
        for item in undirected_results:
            key = tuple(sorted((item['exp_a'], item['exp_b'])))
            score_lookup[key] = {
                's_sem': item['s_sem'],
                's_sem_reason': item.get('s_sem_reason', '')
            }

        directed_results = []
        for item in merged_candidates:
            exp_id = item['exp_id']
            updated_candidates = []

            for cand in item.get('candidates', []):
                cand_id = cand['candidate_id']
                key = tuple(sorted((exp_id, cand_id)))
                sem = score_lookup.get(key, {'s_sem': 0.0, 's_sem_reason': 'Missing semantic score'})

                updated_candidates.append({
                    'candidate_id': cand_id,
                    's_ent': cand['s_ent'],
                    's_graph': cand['s_graph'],
                    's_sem': sem['s_sem'],
                    's_sem_reason': sem['s_sem_reason']
                })

            directed_results.append({
                'exp_id': exp_id,
                'candidates': updated_candidates
            })

        logger.log_info(f"Directed output lists: {len(directed_results)} (aligned with merged/entity candidates)")

        return directed_results

    def _compute_pair_semantic_similarity(
        self,
        exp1: Dict[str, Any],
        exp2: Dict[str, Any]
    ) -> tuple:
        """Compute semantic similarity between two experiences using LLM.

        Args:
            exp1: Experience dict
            exp2: Experience dict

        Returns:
            Tuple of (s_sem score, reason string)
        """
        try:
            # Format prompt
            prompt_vars = {
                'condition_a': exp1.get('condition', ''),
                'content_a': exp1.get('content', ''),
                'condition_b': exp2.get('condition', ''),
                'content_b': exp2.get('content', '')
            }

            # Invoke LLM
            response = self.chain.invoke(prompt_vars)
            response_text = response.content.strip()

            # Parse JSON response
            # Try to extract from markdown code block first
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Try to extract JSON object directly
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = json.loads(response_text)

            # Compatible with both old/new response schemas:
            # - {"semantic_score": ..., "short_reason": ...}
            # - {"similarity": ..., "reason": ...}
            s_sem_raw = result.get('semantic_score', result.get('similarity', 0.0))
            reason = result.get('short_reason', result.get('reason', ''))

            try:
                s_sem = float(s_sem_raw)
            except (TypeError, ValueError):
                s_sem = 0.0

            # Clamp to [0, 1] for stability
            s_sem = max(0.0, min(1.0, s_sem))

            return s_sem, reason

        except Exception as e:
            logger.log_warning(f"Failed to compute semantic similarity: {str(e)}")
            return 0.0, f"Error: {str(e)}"

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save semantic similarity results to JSONL file.

        Args:
            results: List of semantic similarity results
            output_path: Path to save JSONL file
        """
        logger.log_info(f"\nSaving semantic scores to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        logger.log_info(f"Saved {len(results)} semantic score lists")
