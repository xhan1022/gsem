"""Main entry point for GSEM TTL (Test-Time Learning) — Online Self-Evolution.

Usage:
  python main_ttl.py                      # run on full test set
  python main_ttl.py --limit 10           # first 10 cases (quick test)
  python main_ttl.py --resume             # resume from existing TTL checkpoint

Data layout:
  data/experiences/           ← static library (Phase 1+2, 1104 entries, READ-ONLY)
  data/ttl/experiences/       ← TTL dynamic library (new online experiences)
  data/ttl/graph_state.json   ← TTL dynamic graph state (qualities, corrections)
  data/ttl/results.jsonl      ← per-case results

TTL pipeline (10 steps per case):
  Step 1  Inference with retrieved experiences  (new)
  Step 2  Evaluate correctness                  (reuses Phase 1 evaluator)
  Step 3  Extract one experience                (new)
  Step 4  ERV for new experience                (reuses Phase 1 Stage6ERV)
  Step 5  Core entity extraction                (reuses Phase 2)
  Step 6  Entity normalisation                  (reuses Phase 2)
  Step 7  Role-edge extraction                  (reuses Phase 2)
  Step 8  Similarity vs retrieved set           (reuses Phase 2 formulas)
  Step 9  Merge into experience graph           (reuses Phase 2 graph logic)
  Step 10 Online weight update                  (TTL math)
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.shared.config import config
from src.shared.logger import logger
from src.phase3.ttl import (
    GraphState,
    OnlineEvolutionPipeline,
    GSEMRetrievalAdapter,
)
from main import load_cases  # reuse Phase 1 case loader


def main():
    parser = argparse.ArgumentParser(
        description="GSEM TTL — Test-Time Learning (Online Self-Evolution)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N cases (0 = all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing TTL checkpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of experiences to retrieve per case (default: 5)",
    )
    parser.add_argument(
        "--erv-samples",
        type=int,
        default=5,
        help="ERV sampling count per new experience (default: 5)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.35,
        help="Minimum edge weight W to build an edge (default: 0.35)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--ttl-dir",
        type=str,
        default="data/ttl",
        help="TTL output directory (default: data/ttl)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Redirect logger to a TTL-specific log file
    # ------------------------------------------------------------------
    log_path = logger._renew("ttl")

    # ------------------------------------------------------------------
    # Validate config
    # ------------------------------------------------------------------
    try:
        config.validate()
    except ValueError as e:
        logger.log_error(f"Config error: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load test cases (reuse Phase 1 loader, forced to test split)
    # ------------------------------------------------------------------
    os.environ["DATASET_SPLIT"] = "test"

    logger.log_info("\n" + "=" * 80)
    logger.log_info("GSEM TTL — TEST-TIME LEARNING (ONLINE SELF-EVOLUTION)")
    logger.log_info(f"Log file : {log_path}")
    logger.log_info(f"  → tail -f {log_path}")
    logger.log_info("=" * 80)

    cases = load_cases()
    if not cases:
        logger.log_error("No test cases loaded. Check TEST_DATASET_PATH in .env.")
        sys.exit(1)

    if args.limit > 0:
        cases = cases[: args.limit]

    logger.log_info(f"Test cases: {len(cases)}")

    # ------------------------------------------------------------------
    # Load / init TTL graph state
    # ------------------------------------------------------------------
    entity_stats_path = os.path.join(args.data_dir, "entity_stats", "entity_stats.json")
    lexicon_path = os.path.join(args.data_dir, "lexicon", "lexicon.json")

    if args.resume:
        logger.log_info("Resuming from existing TTL checkpoint …")
        graph_state = GraphState.load_or_init(args.data_dir, args.ttl_dir)
    else:
        logger.log_info("Initialising TTL graph state from static Phase 2 outputs …")
        graph_state = GraphState.load_from_static(args.data_dir, args.ttl_dir)

    logger.log_info(
        f"Graph loaded: {len(graph_state.experiences)} experiences "
        f"({sum(1 for e in graph_state.experiences if graph_state.is_ttl(e))} TTL), "
        f"{len(graph_state.adjacency)} nodes with edges"
    )

    # ------------------------------------------------------------------
    # Retrieval interface (stub by default; replace with real implementation)
    # ------------------------------------------------------------------
    retrieval = GSEMRetrievalAdapter(top_k=args.top_k)

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    pipeline = OnlineEvolutionPipeline(
        retrieval_interface=retrieval,
        graph_state=graph_state,
        top_k=args.top_k,
        erv_samples=args.erv_samples,
        edge_threshold=args.edge_threshold,
        entity_stats_path=entity_stats_path,
        lexicon_path=lexicon_path,
        ttl_dir=args.ttl_dir,
    )

    # ------------------------------------------------------------------
    # Process cases
    # ------------------------------------------------------------------
    os.makedirs(args.ttl_dir, exist_ok=True)
    results_path = os.path.join(args.ttl_dir, "results.jsonl")
    results_file = open(results_path, "a", encoding="utf-8")

    results = []
    total = len(cases)

    try:
        for idx, case in enumerate(cases, 1):
            logger.log_info(f"\n[{idx}/{total}] Case: {case.get('case_id', '?')}")
            try:
                result = pipeline.process_case(case)
            except Exception as e:
                logger.log_error(f"Case {case.get('case_id')} failed: {e}")
                import traceback
                traceback.print_exc()
                result = {
                    "case_id": case.get("case_id"),
                    "score": 0,
                    "error": str(e),
                }

            results.append(result)
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_file.flush()

            # Save TTL state after each case
            graph_state.save(args.ttl_dir)

    except KeyboardInterrupt:
        logger.log_warning("\nInterrupted — saving TTL checkpoint …")
        graph_state.save(args.ttl_dir)
    finally:
        results_file.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    scored = [r for r in results if "score" in r]
    accuracy = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
    new_exps = sum(1 for r in results if r.get("new_experience_id"))
    ttl_total = sum(1 for e in graph_state.experiences if graph_state.is_ttl(e))

    logger.log_info("\n" + "=" * 80)
    logger.log_info("TTL COMPLETE")
    logger.log_info("=" * 80)
    logger.log_info(f"  Cases processed        : {len(results)}")
    logger.log_info(f"  Overall accuracy       : {accuracy:.2%}")
    logger.log_info(f"  New TTL experiences    : {new_exps}")
    logger.log_info(f"  Total TTL experiences  : {ttl_total}")
    logger.log_info(
        f"  TTL experience library : {args.ttl_dir}/experiences/new_experiences.jsonl"
    )
    logger.log_info(f"  Results                : {results_path}")
    logger.log_info("=" * 80)


if __name__ == "__main__":
    main()
