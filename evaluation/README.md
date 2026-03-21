# Evaluation Baselines (Cleaned)

本目录当前仅保留两类方法：

- `GSEMAgent`（最终方法）
  - 两阶段初始点召回：`FindStartTwoStage`
  - 多路图遍历检索：`retriver_multi_start`
- `GSEM` 消融方法（`gsem_ablation_agent.py`）
  - `GSEMSemanticOnlyAgent`
  - `GSEMEntityOnlyAgent`
  - `GSEMSinglePathAgent`
  - `GSEMMultiStartNoFillAgent`

已移除：

- `pro/promax` 相关 baseline 与检索器
- `vanilla` 与 `naive_rag` baseline

检索模块路径：

- `src/phase3/retrieval/find_start_two_stage.py`
- `src/phase3/retrieval/retriver_multi_start.py`
- `src/phase3/retrieval/retriver_multi_start_no_fill.py`（消融）
- `src/phase3/retrieval/retriver.py`（单路遍历消融使用）
