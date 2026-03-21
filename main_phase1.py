"""Main entry point for GSEM Phase 1 Pipeline."""
import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.config import config
from src.shared.logger import logger
from src.phase1.pipeline import GSEMPipeline
from src.shared.utils import load_json


def clear_all_outputs():
    """Clear all output directories before each run.

    For research purposes, we always regenerate all outputs to ensure:
    - Consistency with current prompts
    - Reproducibility of experiments
    - No stale cached data
    """
    logger.log_info("清理所有输出目录...")

    # Clear all output directories
    dirs_to_clear = [
        config.paths.trajectories_dir,
        config.paths.evaluations_dir,
        config.paths.normalized_dir,
        config.paths.positive_knowledge_dir,
        config.paths.failure_analysis_dir,
        config.paths.experiences_dir
    ]

    for dir_path in dirs_to_clear:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.log_info(f"  清理: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

    logger.log_info("输出目录清理完成\n")


def normalize_case(raw_case: dict, case_idx: int) -> dict:
    """Normalize case to standard format if needed.

    Expected format: {id, task, options, answer}
    """
    # Already normalized
    if "id" in raw_case and "task" in raw_case and "answer" in raw_case:
        return {
            "case_id": raw_case["id"],
            "description": raw_case["task"],
            "options": raw_case.get("options", {}),
            "answer": raw_case["answer"],
            "task_type": raw_case.get("task_type", ""),
            "reference_analysis": raw_case.get("reference_analysis", "")
        }

    # Old format with case_id, description, gold_standard
    if "case_id" in raw_case and "description" in raw_case:
        return {
            "case_id": raw_case["case_id"],
            "description": raw_case["description"],
            "options": raw_case.get("options", {}),
            "answer": raw_case.get("gold_standard", {}).get("answer", raw_case.get("answer", "")),
            "task_type": raw_case.get("task_type", ""),
            "reference_analysis": raw_case.get("reference_analysis", "")
        }

    # Raw format with question, answer, options
    case_id = raw_case.get("realidx", case_idx)
    if isinstance(case_id, int):
        case_id = f"case_{case_id:04d}"

    question = raw_case.get("question", "")
    options = raw_case.get("options", {})

    if options:
        options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
        task = f"{question}\n\n选项:\n{options_text}"
    else:
        task = question

    return {
        "case_id": case_id,
        "description": task,
        "options": options,
        "answer": raw_case.get("answer", ""),
        "task_type": raw_case.get("task_type", ""),
        "reference_analysis": raw_case.get("reference_analysis", "")
    }


def load_cases() -> list:
    """Load all cases from the dataset configured in config.

    Returns:
        List of clinical cases in normalized format
    """
    dataset_path = config.paths.dataset_path

    if not os.path.exists(dataset_path):
        logger.log_error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load dataset
    try:
        raw_cases = load_json(dataset_path)
        if not isinstance(raw_cases, list):
            logger.log_error(f"Dataset must be a JSON array, got {type(raw_cases)}")
            sys.exit(1)
    except Exception as e:
        logger.log_error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)

    # Normalize all cases
    cases = []
    for idx, raw_case in enumerate(raw_cases, 1):
        try:
            normalized = normalize_case(raw_case, idx)
            cases.append(normalized)
        except Exception as e:
            logger.log_warning(f"Failed to normalize case {idx}: {str(e)}")

    return cases


def main():
    """Main function to run the pipeline."""
    try:
        # Validate configuration
        config.validate()

        # Display dataset configuration
        logger.log_info(f"Dataset split: {config.paths.dataset_split}")
        logger.log_info(f"Dataset path: {config.paths.dataset_path}")

        # Always clear all outputs for reproducibility
        clear_all_outputs()

        # Load cases
        logger.log_info("Loading cases...")
        cases = load_cases()
        logger.log_info(f"Loaded {len(cases)} cases\n")

        # Initialize and run pipeline
        pipeline = GSEMPipeline()
        experience_library = pipeline.run(cases)

        # Print summary
        stats = experience_library.get("statistics", {})
        logger.log_info("\n" + "=" * 80)
        logger.log_info("Experience Library Summary:")
        logger.log_info(f"  Total Experiences: {stats.get('total_experiences', 0)}")
        logger.log_info(f"  - Indication: {stats.get('indication_count', 0)}")
        logger.log_info(f"  - Contraindication: {stats.get('contraindication_count', 0)}")
        logger.log_info(f"  - Strategy: {stats.get('strategy_count', 0)}")
        logger.log_info("=" * 80)

        logger.log_info("\n✓ Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.log_warning("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.log_error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
