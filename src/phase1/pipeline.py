"""GSEM Pipeline orchestration using LangGraph."""
import os
import json
from typing import Dict, Any, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END

from src.shared.config import config
from src.shared.logger import logger
from src.shared.utils import save_json, save_jsonl
from .stages import (
    Stage1Rollout,
    Stage2TrajectoryNormalization,
    Stage3PositiveKnowledgeExtraction,
    Stage4FailureAnalysis,
    Stage5Deduplication,
    Stage6ERV
)


class PipelineState(TypedDict):
    """State for the pipeline graph."""
    cases: List[Dict[str, Any]]
    stage1_results: List[Dict[str, Any]]
    stage2_results: List[Dict[str, Any]]
    stage3_results: List[Dict[str, Any]]
    stage4_results: List[Dict[str, Any]]
    stage5_results: List[Dict[str, Any]]
    experience_library: Dict[str, Any]
    success_count: int
    failed_count: int


class GSEMPipeline:
    """GSEM Phase 1 Pipeline - 6 Stages (with ERV)."""

    def __init__(self):
        """Initialize pipeline with all stages."""
        self.stage1 = Stage1Rollout()
        self.stage2 = Stage2TrajectoryNormalization()
        self.stage3 = Stage3PositiveKnowledgeExtraction()
        self.stage4 = Stage4FailureAnalysis()
        self.stage5 = Stage5Deduplication()
        self.stage6 = Stage6ERV()  # ERV is Stage 6, executed per-case

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline.

        Returns:
            Compiled state graph
        """
        workflow = StateGraph(PipelineState)

        # Add nodes for each stage
        workflow.add_node("stage1", self._run_stage1)
        workflow.add_node("stage2", self._run_stage2)
        workflow.add_node("stage3", self._run_stage3)
        workflow.add_node("stage4_and_5", self._run_stage4_and_5_parallel)  # Parallel execution
        workflow.add_node("stage6", self._run_stage6)
        # Note: Stage 6 ERV is now executed per-case, not as a separate graph node

        # Define edges (sequential flow with parallel stage 4&5)
        workflow.set_entry_point("stage1")
        workflow.add_edge("stage1", "stage2")
        workflow.add_edge("stage2", "stage3")
        workflow.add_edge("stage3", "stage4_and_5")
        workflow.add_edge("stage4_and_5", "stage6")
        workflow.add_edge("stage6", END)

        return workflow.compile()

    def _run_stage1(self, state: PipelineState) -> PipelineState:
        """Run Stage 1: Trajectory Generation."""
        results = self.stage1.run(state["cases"])
        state["stage1_results"] = results
        return state

    def _run_stage2(self, state: PipelineState) -> PipelineState:
        """Run Stage 2: Trajectory Normalization."""
        results = self.stage2.run(state["stage1_results"])
        state["stage2_results"] = results
        return state

    def _run_stage3(self, state: PipelineState) -> PipelineState:
        """Run Stage 3: Positive Knowledge & Failure Analysis."""
        # This is now handled by stage4_and_5_parallel
        state["stage3_results"] = state["stage2_results"]
        return state

    def _run_stage4_and_5_parallel(self, state: PipelineState) -> PipelineState:
        """Run Stage 3 and 4 in parallel (Positive Knowledge + Failure Analysis)."""
        logger.log_info("\n并行执行阶段3和阶段4...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both stages
            future_stage3 = executor.submit(self.stage3.run, state["stage3_results"])
            future_stage4 = executor.submit(self.stage4.run, state["stage3_results"])

            # Wait for both to complete
            stage3_results = future_stage3.result()
            stage4_results = future_stage4.result()

        state["stage4_results"] = stage3_results
        state["stage5_results"] = stage4_results

        # Log combined experience counts
        total_indication = sum(r.get("count", 0) for r in stage3_results)
        total_contraindication = sum(r.get("contraindication_count", 0) for r in stage4_results)

        logger.log_info(f"\n阶段3&4完成，提取经验统计:")
        logger.log_info(f"  Indication: {total_indication} 条")
        logger.log_info(f"  Contraindication: {total_contraindication} 条")

        return state

    def _run_stage6(self, state: PipelineState) -> PipelineState:
        """Run Stage 5: Deduplication."""
        experience_library = self.stage5.run(
            state["stage4_results"],
            state["stage5_results"]
        )
        state["experience_library"] = experience_library

        # Calculate success/failure counts
        success_count = len([r for r in state["stage3_results"] if r.get("success_count", 0) > 0])
        failed_count = len(state["cases"]) - success_count

        state["success_count"] = success_count
        state["failed_count"] = failed_count

        return state

    def _process_single_case(self, case: Dict[str, Any], case_num: int, total_cases: int) -> Dict[str, Any]:
        """Process a single case through all 6 stages.

        Args:
            case: Clinical case
            case_num: Current case number
            total_cases: Total number of cases

        Returns:
            Dictionary with all stage results
        """
        try:
            # Stage 1: Rollout (Generation + Evaluation)
            stage1_result = self.stage1.process_case(case, case_num, total_cases)

            # Stage 2: Trajectory Normalization
            stage2_result = self.stage2.process_case(stage1_result)

            # Stage 3 & 4: Parallel execution (Positive Knowledge + Failure Analysis)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_stage3 = executor.submit(self.stage3.process_case, stage2_result)
                future_stage4 = executor.submit(self.stage4.process_case, stage2_result)

                stage3_result = future_stage3.result()
                stage4_result = future_stage4.result()

            # Log experience extraction for this case
            indication_count = stage3_result.get("count", 0)
            contraindication_count = stage4_result.get("contraindication_count", 0)

            logger.log_experiences(
                indication_count,
                contraindication_count
            )

            # Collect all experiences for this case
            all_case_experiences = []
            all_case_experiences.extend(stage3_result.get("indications", []))
            all_case_experiences.extend(stage4_result.get("contraindications", []))

            # Stage 5: Deduplication (per-case)
            if all_case_experiences:
                logger.log_info(f"  [Stage 5] 单案例去重: {len(all_case_experiences)} 条经验")
                deduplicated_experiences = self.stage5.deduplicate_case_experiences(all_case_experiences)
                logger.log_info(f"  [Stage 5] 去重后: {len(deduplicated_experiences)} 条")

                # Update results with deduplicated experiences
                dedup_indications = [exp for exp in deduplicated_experiences if exp.get("type") == "Indication"]
                dedup_contraindications = [exp for exp in deduplicated_experiences if exp.get("type") == "Contraindication"]

                stage3_result["indications"] = dedup_indications
                stage3_result["count"] = len(dedup_indications)
                stage4_result["contraindications"] = dedup_contraindications
                stage4_result["contraindication_count"] = len(dedup_contraindications)

                # Stage 6: ERV Validation (per-case)
                baseline_rate = stage1_result.get("success_count", 0) / stage1_result.get("sampling_count", config.pipeline.sampling_count)
                logger.log_info(f"  [Stage 6] 开始ERV验证 (baseline_rate={baseline_rate:.2f})")

                erv_result = self.stage6.validate_case_experiences(
                    case,
                    deduplicated_experiences,
                    baseline_rate
                )

                # Attach quality to experiences
                quality = erv_result["quality"]
                self.stage6.attach_quality_to_experiences(dedup_indications, quality)
                self.stage6.attach_quality_to_experiences(dedup_contraindications, quality)

                logger.log_info(f"  [Stage 6] Quality分数: {quality}")
            else:
                logger.log_info(f"  [Stage 5] 跳过（无经验可去重）")
                logger.log_info(f"  [Stage 6] 跳过（无经验可验证）")

            return {
                "stage1": stage1_result,
                "stage3": stage3_result,
                "stage4": stage4_result,
                "case_id": case.get("case_id")
            }

        except Exception as e:
            logger.log_error(f"Case processing failed: {str(e)}", case.get("case_id"))
            return None

    def _append_experiences_to_file(self, stage3_result: Dict[str, Any], stage4_result: Dict[str, Any]):
        """Append experiences to the experience library file.

        Args:
            stage3_result: Stage 3 results with Indication experiences
            stage4_result: Stage 4 results with Contraindication experiences
        """
        experiences_file = os.path.join(config.paths.experiences_dir, "experiences_raw.jsonl")

        # Collect all experiences
        all_experiences = []

        # Add Indication experiences
        if stage3_result and "indications" in stage3_result:
            all_experiences.extend(stage3_result["indications"])

        # Add Contraindication experiences
        if stage4_result and "contraindications" in stage4_result:
            all_experiences.extend(stage4_result["contraindications"])

        # Append to file
        if all_experiences:
            with open(experiences_file, "a", encoding="utf-8") as f:
                for exp in all_experiences:
                    f.write(json.dumps(exp, ensure_ascii=False) + "\n")

    def run(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the complete pipeline: each case goes through all 6 stages.

        Args:
            cases: List of clinical cases

        Returns:
            Final experience library with quality scores
        """
        logger.start_pipeline(len(cases))

        # Clear previous raw experiences file
        experiences_file = os.path.join(config.paths.experiences_dir, "experiences_raw.jsonl")
        if os.path.exists(experiences_file):
            os.remove(experiences_file)

        total_cases = len(cases)
        stage3_results = []
        stage4_results = []

        # Process each case through all 6 stages
        for idx, case in enumerate(cases, 1):
            result = self._process_single_case(case, idx, total_cases)

            if result:
                stage3_results.append(result["stage3"])
                stage4_results.append(result["stage4"])

                # Append experiences to file immediately (already deduplicated and with quality)
                self._append_experiences_to_file(result["stage3"], result["stage4"])

        # Collect all experiences
        logger.log_info("\n" + "=" * 80)
        logger.log_info("收集所有案例的经验")
        logger.log_info("=" * 80)
        experience_library = self.stage5.run(stage3_results, stage4_results)

        # Calculate statistics
        success_count = len([r for r in stage3_results if r.get("count", 0) > 0])
        failed_count = total_cases - success_count
        total_experiences = experience_library.get("statistics", {}).get("total_experiences", 0)

        logger.finish_pipeline(success_count, failed_count, total_experiences)

        return experience_library
