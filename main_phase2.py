"""GSEM Experience Graph Construction - Modular Pipeline.

This script runs the complete graph construction pipeline with modular execution:
- Step 1: Preprocessing (ID assignment + task type normalization)
- Step 2: Core entity extraction with roles
- Step 3: Role-edge structure extraction
- Step 4: Entity statistics
- Step 5a: Entity candidate retrieval + S_ent
- Step 5b: Structure candidate retrieval + S_graph
- Step 5c: Semantic similarity computation
- Step 5d: Task similarity computation
- Step 6: Graph construction
- Step 7: NetworkX export

Usage:
  # Run all steps
  python main_graph.py

  # Run specific steps
  python main_graph.py --steps 1
  python main_graph.py --steps 2
  python main_graph.py --steps 2,3,4

  # Resume from a specific step
  python main_graph.py --from 3
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.shared.config import config
from src.shared.logger import logger
from src.phase2.graph.preprocessing import ExperiencePreprocessor
from src.phase2.graph.entity_extraction import CoreEntityExtractor
from src.phase2.graph.entity_normalizer import normalize_jsonl
from src.phase2.graph.structure_extraction import RoleEdgeExtractor
from src.phase2.graph.entity_stats import EntityStats
from src.phase2.graph.entity_candidate_retriever import EntityCandidateRetriever
from src.phase2.graph.structure_candidate_retriever import StructureCandidateRetriever
from src.phase2.graph.semantic_similarity_computer import SemanticSimilarityComputer
from src.phase2.graph.task_similarity_computer import TaskSimilarityComputer
from src.phase2.graph.export_to_networkx import NetworkXExporter


def load_results(file_path: str):
    """Load results from JSONL file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_experiences(experiences_path: str):
    """Load experiences from Phase 1 output."""
    logger.log_info(f"Loading experiences from: {experiences_path}")

    if not os.path.exists(experiences_path):
        logger.log_error(f"Experience file not found: {experiences_path}")
        return []

    experiences = []
    with open(experiences_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                experiences.append(json.loads(line))

    logger.log_info(f"Loaded {len(experiences)} experiences")
    return experiences


def run_step1(experiences, output_dir):
    """Step 1: Preprocessing (ID assignment + task type normalization)."""
    preprocessor = ExperiencePreprocessor()
    experiences = preprocessor.process(experiences)

    # Save preprocessed experiences
    preprocessed_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    preprocessed_path = os.path.join(preprocessed_dir, "experiences.jsonl")
    preprocessor.save_results(experiences, preprocessed_path)

    logger.log_info(f"  Preprocessed experiences saved: {preprocessed_path}")

    return experiences


def run_step2(experiences, output_dir):
    """Step 2: Core entity extraction with roles (extraction only, no normalization)."""
    extractor = CoreEntityExtractor()

    # Incremental output
    output_path = os.path.join(output_dir, "core_entities/core_entities.jsonl")
    results = extractor.process_experiences(
        experiences,
        output_path=output_path,
        incremental=True
    )

    logger.log_info(f"\nStep 2 Complete:")
    logger.log_info(f"  Core entities saved: {output_path}")
    logger.log_info(f"  Note: Run Step 2.5 for normalization")

    return results


def run_step2_5(output_dir):
    """Step 2.5: Entity normalization (Form + Semantic)."""
    input_path = os.path.join(output_dir, "core_entities/core_entities.jsonl")
    lexicon_path = os.path.join(output_dir, "lexicon/lexicon.json")

    # Call normalization function (logging is done inside)
    normalize_jsonl(input_path, lexicon_path)

    # Reload updated results
    results = load_results(input_path)
    return results


def run_step3(entity_results, output_dir):
    """Step 3: Role-edge structure extraction."""
    extractor = RoleEdgeExtractor()

    # Incremental output
    output_path = os.path.join(output_dir, "role_edges/role_edges.jsonl")
    results = extractor.process_experiences(
        entity_results,
        output_path=output_path,
        incremental=True
    )

    logger.log_info(f"\nStep 3 Complete:")
    logger.log_info(f"  Role-edges saved: {output_path}")

    return results


def run_step4(structure_results, output_dir):
    """Step 4: Entity statistics and IDF computation."""
    logger.log_info("\n" + "=" * 80)
    logger.log_info("STEP 4: Entity Statistics and IDF Computation")
    logger.log_info("=" * 80)

    stats_computer = EntityStats()
    stats = stats_computer.compute_stats(structure_results)
    postings = stats_computer.get_postings()

    # Save results
    stats_dir = os.path.join(output_dir, "entity_stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "entity_stats.json")

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    postings_dir = os.path.join(output_dir, "entity_postings")
    os.makedirs(postings_dir, exist_ok=True)
    postings_path = os.path.join(postings_dir, "entity_postings.json")

    with open(postings_path, 'w', encoding='utf-8') as f:
        json.dump(postings, f, ensure_ascii=False, indent=2)

    logger.log_info(f"\nStep 4 Complete:")
    logger.log_info(f"  Entity stats saved: {stats_path}")
    logger.log_info(f"  Entity postings saved: {postings_path}")
    logger.log_info(f"  Unique entities: {len(stats['df'])}")

    return stats, postings


def run_step5a(structure_results, entity_stats_path, entity_postings_path, output_dir):
    """Step 5a: Entity candidate retrieval + S_ent."""
    logger.log_info("\n" + "=" * 80)
    logger.log_info("STEP 5a: Entity Candidate Retrieval + S_ent")
    logger.log_info("=" * 80)

    retriever = EntityCandidateRetriever(
        entity_stats_path=entity_stats_path,
        entity_postings_path=entity_postings_path,
        top_k=60
    )

    candidates = retriever.retrieve_candidates(structure_results)

    # Save results
    candidates_dir = os.path.join(output_dir, "entity_candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    candidates_path = os.path.join(candidates_dir, "entity_candidates.jsonl")
    retriever.save_results(candidates, candidates_path)

    logger.log_info(f"\nStep 5a Complete:")
    logger.log_info(f"  Entity candidates saved: {candidates_path}")

    return candidates


def run_step5b(structure_results, output_dir):
    """Step 5b: Structure candidate retrieval + S_graph."""
    logger.log_info("\n" + "=" * 80)
    logger.log_info("STEP 5b: Structure Candidate Retrieval + S_graph")
    logger.log_info("=" * 80)

    # Toggle path-token mode:
    # - "hybrid": S1 role+entity, S2/3/4 role-only (current default)
    # - "role_only": legacy role-only for all S1~S4
    # - "entity_aware": role+entity strict exact match for all S1~S4
    # - "entity_aware_relaxed": role+entity for all S1~S4, overlap>=ceil(n/2)
    # - "entity_aware_relaxed_s1_role_only": S1 role-only; S2/3/4 role+entity relaxed
    path_token_mode = "entity_aware_relaxed"

    lexicon_path = os.path.join(output_dir, "lexicon", "lexicon.json")
    retriever = StructureCandidateRetriever(
        top_k=40,
        lexicon_path=lexicon_path,
        path_token_mode=path_token_mode
    )

    # Path to entity candidates
    entity_candidates_path = os.path.join(output_dir, "entity_candidates", "entity_candidates.jsonl")

    candidates = retriever.retrieve_candidates(structure_results, entity_candidates_path)

    # Save results
    candidates_dir = os.path.join(output_dir, "structure_candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    candidates_path = os.path.join(candidates_dir, "structure_candidates.jsonl")
    retriever.save_results(candidates, candidates_path)

    logger.log_info(f"\nStep 5b Complete:")
    logger.log_info(f"  Structure candidates saved: {candidates_path}")

    return candidates


def run_step5c(entity_candidates, structure_candidates, structure_results, output_dir):
    """Step 5c: Semantic similarity computation."""
    logger.log_info("\n" + "=" * 80)
    logger.log_info("STEP 5c: Semantic Similarity Computation")
    logger.log_info("=" * 80)

    computer = SemanticSimilarityComputer(
        model_name=config.deepseek.model_name,
        api_base=config.deepseek.base_url,
        api_key=config.deepseek.api_key,
        merged_top_k=100,
        temperature=0.0
    )

    # Merge candidates
    merged_candidates = computer.merge_candidates(entity_candidates, structure_candidates)

    # Compute semantic similarity
    results = computer.compute_semantic_similarity(merged_candidates, structure_results)

    # Save results
    scores_dir = os.path.join(output_dir, "semantic_scores")
    os.makedirs(scores_dir, exist_ok=True)
    scores_path = os.path.join(scores_dir, "semantic_scores.jsonl")
    computer.save_results(results, scores_path)

    logger.log_info(f"\nStep 5c Complete:")
    logger.log_info(f"  Semantic scores saved: {scores_path}")

    return results


def run_step5d(semantic_results, structure_results, output_dir):
    """Step 5d: Task similarity computation."""
    logger.log_info("\n" + "=" * 80)
    logger.log_info("STEP 5d: Task Similarity Computation")
    logger.log_info("=" * 80)

    computer = TaskSimilarityComputer()

    # Compute task similarity
    results = computer.compute_task_similarity(semantic_results, structure_results)

    # Save results
    scores_dir = os.path.join(output_dir, "task_scores")
    os.makedirs(scores_dir, exist_ok=True)
    scores_path = os.path.join(scores_dir, "task_scores.jsonl")
    computer.save_results(results, scores_path)

    logger.log_info(f"\nStep 5d Complete:")
    logger.log_info(f"  Task scores saved: {scores_path}")

    return results


def run_step6(task_results, output_dir, alpha=0.25, beta=0.25, gamma=0.4, delta=0.1, threshold=0.35):
    """Step 6: Graph construction with edge weight fusion."""
    logger.log_info("\n" + "=" * 80)
    logger.log_info("STEP 6: Edge Weight Fusion and Graph Construction")
    logger.log_info("=" * 80)
    logger.log_info(f"Weight parameters: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    logger.log_info(f"Edge threshold: W >= {threshold}")

    # Build edges
    edges = []

    # New format: undirected unique pairs from Step 5d
    if task_results and 'exp_a' in task_results[0] and 'exp_b' in task_results[0]:
        logger.log_info("Input format: undirected unique pairs")
        for item in task_results:
            src_id = item['exp_a']
            dst_id = item['exp_b']
            s_ent = item['s_ent']
            s_graph = item['s_graph']
            s_sem = item['s_sem']
            s_task = item['s_task']

            W = alpha * s_ent + beta * s_graph + gamma * s_sem + delta * s_task

            if W >= threshold:
                edges.append({
                    'src': src_id,
                    'dst': dst_id,
                    'S_ent': s_ent,
                    'S_graph': s_graph,
                    'S_sem': s_sem,
                    'S_task': s_task,
                    'W': W,
                    'short_reason': item.get('s_sem_reason', '')
                })

    # Legacy format: directed candidate lists (backward compatibility)
    else:
        logger.log_info("Input format: legacy directed pairs")
        seen_pairs = set()
        for item in task_results:
            src_id = item['exp_id']
            for cand in item['candidates']:
                dst_id = cand['candidate_id']
                if src_id == dst_id:
                    continue

                pair_key = tuple(sorted((src_id, dst_id)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                s_ent = cand['s_ent']
                s_graph = cand['s_graph']
                s_sem = cand['s_sem']
                s_task = cand['s_task']

                W = alpha * s_ent + beta * s_graph + gamma * s_sem + delta * s_task

                if W >= threshold:
                    edges.append({
                        'src': src_id,
                        'dst': dst_id,
                        'S_ent': s_ent,
                        'S_graph': s_graph,
                        'S_sem': s_sem,
                        'S_task': s_task,
                        'W': W,
                        'short_reason': cand.get('s_sem_reason', '')
                    })

    # Build adjacency list
    adjacency_list = defaultdict(list)
    for edge in edges:
        src = edge['src']
        dst = edge['dst']

        adjacency_list[src].append({
            'neighbor': dst,
            'W': edge['W'],
            'S_ent': edge['S_ent'],
            'S_graph': edge['S_graph'],
            'S_sem': edge['S_sem'],
            'S_task': edge['S_task']
        })

        adjacency_list[dst].append({
            'neighbor': src,
            'W': edge['W'],
            'S_ent': edge['S_ent'],
            'S_graph': edge['S_graph'],
            'S_sem': edge['S_sem'],
            'S_task': edge['S_task']
        })

    adjacency_list = dict(adjacency_list)

    # Save results
    edges_dir = os.path.join(output_dir, "experience_edges")
    os.makedirs(edges_dir, exist_ok=True)
    edges_path = os.path.join(edges_dir, "experience_edges.jsonl")
    with open(edges_path, "w", encoding="utf-8") as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    adj_dir = os.path.join(output_dir, "experience_graph_adj")
    os.makedirs(adj_dir, exist_ok=True)
    adj_path = os.path.join(adj_dir, "experience_graph_adj.jsonl")
    with open(adj_path, "w", encoding="utf-8") as f:
        for node_id in sorted(adjacency_list.keys()):
            neighbors_sorted = sorted(
                adjacency_list[node_id],
                key=lambda x: float(x.get("W", 0.0)),
                reverse=True
            )
            row = {
                "exp_id": node_id,
                "neighbors": neighbors_sorted
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    num_nodes = len(adjacency_list)
    avg_degree = sum(len(neighbors) for neighbors in adjacency_list.values()) / num_nodes if num_nodes > 0 else 0

    logger.log_info(f"\nStep 6 Complete:")
    logger.log_info(f"  Edges saved: {edges_path}")
    logger.log_info(f"  Adjacency list (jsonl) saved: {adj_path}")
    logger.log_info(f"  Graph nodes: {num_nodes}")
    logger.log_info(f"  Graph edges: {len(edges)}")
    logger.log_info(f"  Average degree: {avg_degree:.2f}")

    return edges, adjacency_list


def run_step7(structure_results, edges, output_dir):
    """Step 7: Export graphs to NetworkX formats."""
    logger.log_info("\n" + "=" * 80)
    logger.log_info("STEP 7: Export to NetworkX Formats")
    logger.log_info("=" * 80)

    # Enrich structure_results with quality from preprocessed (quality is not
    # propagated through Steps 2-3, but is preserved in preprocessed/experiences.jsonl)
    preprocessed_path = os.path.join(output_dir, "preprocessed/experiences.jsonl")
    if os.path.exists(preprocessed_path):
        quality_map = {}
        with open(preprocessed_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if 'id' in rec and 'quality' in rec:
                        quality_map[rec['id']] = rec['quality']
        for exp in structure_results:
            if 'quality' not in exp and exp.get('id') in quality_map:
                exp['quality'] = quality_map[exp['id']]
        logger.log_info(f"  Enriched {len(quality_map)} experiences with quality scores")

    nx_dir = os.path.join(output_dir, "networkx")
    exporter = NetworkXExporter(output_dir=nx_dir)

    exporter.export_entity_graph(structure_results, output_prefix="entity_graph")
    exporter.export_experience_graph(structure_results, edges, output_prefix="experience_graph")

    logger.log_info(f"\nStep 7 Complete:")
    logger.log_info(f"  NetworkX files saved to: {nx_dir}")


def main():
    """Run GSEM graph construction pipeline with modular execution."""
    # Force unbuffered output for real-time logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="GSEM Experience Graph Construction - Modular Pipeline")
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run (e.g., "1,2a,2b,2c,3" or "all")'
    )
    parser.add_argument(
        '--from',
        type=str,
        dest='from_step',
        help='Resume from this step (e.g., "3")'
    )

    args = parser.parse_args()

    # Define step order
    all_steps = ['1', '2', '2.5', '3', '4', '5a', '5b', '5c', '5d', '6', '7']

    # Determine which steps to run
    if args.from_step:
        if args.from_step not in all_steps:
            logger.log_error(f"Invalid step: {args.from_step}")
            return
        start_idx = all_steps.index(args.from_step)
        steps_to_run = all_steps[start_idx:]
    elif args.steps == 'all':
        steps_to_run = all_steps
    else:
        steps_to_run = [s.strip() for s in args.steps.split(',')]

    logger.log_info("\n" + "=" * 100)
    logger.log_info("GSEM EXPERIENCE GRAPH CONSTRUCTION - MODULAR PIPELINE")
    logger.log_info("=" * 100)
    logger.log_info(f"Steps to run: {', '.join(steps_to_run)}")
    logger.log_info("=" * 100)

    # Configuration
    experiences_path = "data/experiences/experiences.jsonl"
    output_dir = "data"

    # Load experiences
    experiences = load_experiences(experiences_path)
    if not experiences:
        logger.log_error("No experiences loaded. Exiting.")
        return

    logger.log_info(f"\nTotal experiences to process: {len(experiences)}")

    # State variables
    entity_results = None
    structure_results = None
    entity_candidates = None
    structure_candidates = None
    semantic_results = None
    task_results = None

    # Run steps
    for step in steps_to_run:
        if step == '1':
            experiences = run_step1(experiences, output_dir)

        elif step == '2':
            # Load preprocessed if needed
            if 'id' not in experiences[0]:
                preprocessed_path = os.path.join(output_dir, "preprocessed/experiences.jsonl")
                if os.path.exists(preprocessed_path):
                    experiences = load_experiences(preprocessed_path)
                else:
                    # Run step 1 automatically
                    experiences = run_step1(experiences, output_dir)
            entity_results = run_step2(experiences, output_dir)

        elif step == '2.5':
            entity_results = run_step2_5(output_dir)

        elif step == '3':
            if entity_results is None:
                # Load from file
                step2_path = os.path.join(output_dir, "core_entities/core_entities.jsonl")
                entity_results = load_results(step2_path)
            structure_results = run_step3(entity_results, output_dir)

        elif step == '4':
            if structure_results is None:
                # Load from file
                step3_path = os.path.join(output_dir, "role_edges/role_edges.jsonl")
                structure_results = load_results(step3_path)
            run_step4(structure_results, output_dir)

        elif step == '5a':
            if structure_results is None:
                step3_path = os.path.join(output_dir, "role_edges/role_edges.jsonl")
                structure_results = load_results(step3_path)
            entity_stats_path = os.path.join(output_dir, "entity_stats/entity_stats.json")
            entity_postings_path = os.path.join(output_dir, "entity_postings/entity_postings.json")
            entity_candidates = run_step5a(structure_results, entity_stats_path, entity_postings_path, output_dir)

        elif step == '5b':
            if structure_results is None:
                step3_path = os.path.join(output_dir, "role_edges/role_edges.jsonl")
                structure_results = load_results(step3_path)
            structure_candidates = run_step5b(structure_results, output_dir)

        elif step == '5c':
            if entity_candidates is None:
                entity_cand_path = os.path.join(output_dir, "entity_candidates/entity_candidates.jsonl")
                entity_candidates = load_results(entity_cand_path)
            if structure_candidates is None:
                struct_cand_path = os.path.join(output_dir, "structure_candidates/structure_candidates.jsonl")
                structure_candidates = load_results(struct_cand_path)
            if structure_results is None:
                step3_path = os.path.join(output_dir, "role_edges/role_edges.jsonl")
                structure_results = load_results(step3_path)
            semantic_results = run_step5c(entity_candidates, structure_candidates, structure_results, output_dir)

        elif step == '5d':
            if semantic_results is None:
                sem_scores_path = os.path.join(output_dir, "semantic_scores/semantic_scores.jsonl")
                semantic_results = load_results(sem_scores_path)
            if structure_results is None:
                step3_path = os.path.join(output_dir, "role_edges/role_edges.jsonl")
                structure_results = load_results(step3_path)
            task_results = run_step5d(semantic_results, structure_results, output_dir)

        elif step == '6':
            if task_results is None:
                task_scores_path = os.path.join(output_dir, "task_scores/task_scores.jsonl")
                task_results = load_results(task_scores_path)
            edges, adjacency_list = run_step6(task_results, output_dir)

        elif step == '7':
            if structure_results is None:
                step3_path = os.path.join(output_dir, "role_edges/role_edges.jsonl")
                structure_results = load_results(step3_path)
            edges_path = os.path.join(output_dir, "experience_edges/experience_edges.jsonl")
            edges = load_results(edges_path)
            run_step7(structure_results, edges, output_dir)

        else:
            logger.log_warning(f"Unknown step: {step}")

    # Final summary
    logger.log_info("\n" + "=" * 100)
    logger.log_info("PIPELINE EXECUTION COMPLETED")
    logger.log_info("=" * 100)
    logger.log_info(f"Steps executed: {', '.join(steps_to_run)}")
    logger.log_info("=" * 100)


if __name__ == "__main__":
    main()
