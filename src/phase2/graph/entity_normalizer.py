#!/usr/bin/env python3
"""
实体统一归一化（形式 + 语义）
输入：core_entities.jsonl（每行含 core_entities 字段）
输出：更新 core_entities.jsonl，每行新增 canonical_entities 字段
      生成 data/lexicon/lexicon.json

流程：
  1. 收集所有唯一实体
  2. 形式归一化（字符串清洗）
  3. Embedding聚类 + LLM精判语义等价
  4. 生成 lexicon
  5. 应用到所有经验
"""

import json
import os
import re
import sys
import argparse
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.shared.config import config
from src.shared.logger import logger

# ─────────────────────────── 配置 ───────────────────────────

SIMILARITY_THRESHOLD = 0.85  # 相似度阈值，用于聚类

# ─────────────────────────── 形式归一化 ───────────────────────────

def normalize_form(text: str) -> str:
    """形式归一化：小写、去空格、去标点等"""
    if not text:
        return ""
    # 转小写
    text = text.lower()
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾标点和空格
    text = text.strip('.,;:!?()[]{}"\' ')
    # 归一化连字符
    text = re.sub(r'\s*-\s*', '-', text)
    return text

# ─────────────────────────── Embedding ───────────────────────────

def get_embeddings(texts: list[str]) -> np.ndarray:
    """批量获取文本 embedding，返回 shape=(N, dim) 的 numpy 数组"""
    all_vectors = []
    batch_size = 10  # DashScope max batch size

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]

        # Prepare request
        headers = {
            "Authorization": f"Bearer {config.embedding.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": config.embedding.model_name,
            "input": batch
        }

        # Call API
        response = requests.post(
            f"{config.embedding.base_url}/embeddings",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Embedding API error: {response.status_code} - {response.text}")

        result = response.json()
        vectors = [item["embedding"] for item in result["data"]]
        all_vectors.extend(vectors)

        # Log progress
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
            logger.log_info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} entities...")

    return np.array(all_vectors, dtype=np.float32)

def cosine_similarity_matrix(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵"""
    v1 = vecs1 / (np.linalg.norm(vecs1, axis=1, keepdims=True) + 1e-10)
    v2 = vecs2 / (np.linalg.norm(vecs2, axis=1, keepdims=True) + 1e-10)
    return v1 @ v2.T

# ─────────────────────────── 语义归一化 ───────────────────────────

def cluster_entities(entities: list[str], embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """基于embedding相似度聚类实体，返回聚类结果（每个聚类是实体索引列表）"""
    if len(entities) == 1:
        return [[0]]

    # 使用层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,  # distance = 1 - cosine_similarity
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # 组织聚类结果
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return list(clusters.values())

def llm_merge_clusters(entity_groups: list[list[str]]) -> dict[str, str]:
    """
    对每个聚类，用LLM判断是否语义等价，选择canonical form。
    返回 {entity: canonical}
    """
    # 初始化 LLM
    llm = ChatOpenAI(
        model_name=config.deepseek.model_name,
        openai_api_base=config.deepseek.base_url,
        openai_api_key=config.deepseek.api_key,
        temperature=0.0
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a biomedical entity normalization expert.

Task: Given the following entities that are semantically similar (by embedding), determine:
1. Are they truly semantically equivalent (synonyms, abbreviations, variants)?
2. If yes, which one should be the canonical form (prefer full form over abbreviation)?
3. If no, treat them as separate entities.

Respond ONLY with valid JSON in this exact format:
{{
  "equivalent": true/false,
  "canonical": "<canonical form if equivalent, else null>",
  "mapping": {{
    "<entity1>": "<canonical>",
    "<entity2>": "<canonical>",
    ...
  }}
}}

Rules:
1. If equivalent=true, all entities map to the same canonical.
2. If equivalent=false, each entity maps to itself.
3. Prefer full medical terms over abbreviations as canonical.
4. Output ONLY the JSON, no extra text."""),
        ("human", "Entities: {entities}")
    ])

    parser = JsonOutputParser()
    chain = prompt_template | llm | parser

    result_mapping = {}
    total_groups = len(entity_groups)
    processed = 0

    for group in entity_groups:
        if len(group) == 1:
            # 单个实体，自己是canonical
            result_mapping[group[0]] = group[0]
            processed += 1
            continue

        # 构建实体列表字符串
        entities_str = json.dumps(group)

        try:
            result = chain.invoke({"entities": entities_str})

            if result.get("equivalent", False):
                canonical = result.get("canonical", group[0])
                mapping = result.get("mapping", {})
                logger.log_info(f"  ✓ Merged: {group} → '{canonical}'")
                for e in group:
                    result_mapping[e] = mapping.get(e, canonical)
            else:
                logger.log_info(f"  ✗ Not equivalent: {group} → separate entities")
                mapping = result.get("mapping", {})
                for e in group:
                    result_mapping[e] = mapping.get(e, e)

        except Exception as ex:
            logger.log_warning(f"LLM error for group {group}: {ex}")
            for e in group:
                result_mapping[e] = e

        processed += 1
        if processed % 10 == 0 or processed == total_groups:
            logger.log_info(f"  Progress: {processed}/{total_groups} groups processed")

    return result_mapping

# ─────────────────────────── 主流程 ───────────────────────────

def normalize_jsonl(
    input_path: str,
    lexicon_path: str = "data/lexicon/lexicon.json",
    similarity_threshold: float = SIMILARITY_THRESHOLD
):
    input_path = Path(input_path)
    lexicon_path = Path(lexicon_path)

    logger.log_info("\n" + "="*80)
    logger.log_info("STEP 2.5: Entity Normalization (Form + Semantic)")
    logger.log_info("="*80 + "\n")

    # ── 1. 收集所有实体 ──
    logger.log_info("Step 1: Collecting all entities...")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    all_raw_entities = []
    entity_to_records = defaultdict(list)  # entity -> [record_idx, ...]

    for idx, line in enumerate(lines):
        record = json.loads(line)
        core_entities = record.get("core_entities", [])
        for ent_dict in core_entities:
            entity = ent_dict.get("entity", "")
            if entity:
                all_raw_entities.append(entity)
                entity_to_records[entity].append(idx)

    unique_raw_entities = list(set(all_raw_entities))
    logger.log_info(f"  Total raw entities: {len(all_raw_entities)}")
    logger.log_info(f"  Unique raw entities: {len(unique_raw_entities)}\n")

    # ── 2. 形式归一化 ──
    logger.log_info("Step 2: Form normalization...")
    raw_to_form = {}
    form_to_raws = defaultdict(set)

    for entity in unique_raw_entities:
        form_normalized = normalize_form(entity)
        raw_to_form[entity] = form_normalized
        form_to_raws[form_normalized].add(entity)

    unique_form_entities = list(form_to_raws.keys())
    logger.log_info(f"  After form normalization: {len(unique_form_entities)} unique entities")
    logger.log_info(f"  Compression ratio: {len(unique_raw_entities) / len(unique_form_entities):.2f}x\n")

    # ── 3. 获取 Embedding ──
    logger.log_info("Step 3: Computing embeddings...")
    form_embeddings = get_embeddings(unique_form_entities)
    logger.log_info(f"  Computed embeddings for {len(unique_form_entities)} entities\n")

    # ── 4. 聚类 ──
    logger.log_info(f"Step 4: Clustering by similarity (threshold={similarity_threshold})...")
    clusters = cluster_entities(unique_form_entities, form_embeddings, similarity_threshold)
    logger.log_info(f"  Found {len(clusters)} clusters")

    # 打印部分聚类
    multi_clusters = [c for c in clusters if len(c) > 1]
    logger.log_info(f"  Multi-entity clusters: {len(multi_clusters)}")
    for i, cluster_indices in enumerate(multi_clusters[:5], 1):
        entities_in_cluster = [unique_form_entities[idx] for idx in cluster_indices]
        logger.log_info(f"    Cluster {i}: {entities_in_cluster}")
    if len(multi_clusters) > 5:
        logger.log_info(f"    ... and {len(multi_clusters) - 5} more\n")
    else:
        logger.log_info("")

    # ── 5. LLM 精判语义等价 ──
    logger.log_info("Step 5: LLM semantic validation...")
    entity_groups = []
    for cluster_indices in clusters:
        entities_in_cluster = [unique_form_entities[idx] for idx in cluster_indices]
        entity_groups.append(entities_in_cluster)

    form_to_canonical = llm_merge_clusters(entity_groups)

    # 统计
    unique_canonicals = len(set(form_to_canonical.values()))
    logger.log_info(f"\n  Final canonical entities: {unique_canonicals}")
    logger.log_info(f"  Semantic compression ratio: {len(unique_form_entities) / unique_canonicals:.2f}x\n")

    # ── 6. 构建完整映射：raw → form → canonical ──
    logger.log_info("Step 6: Building lexicon...")
    canonical_to_aliases = defaultdict(set)
    alias_to_canonical = {}

    for raw_entity in unique_raw_entities:
        form_entity = raw_to_form[raw_entity]
        canonical = form_to_canonical.get(form_entity, form_entity)

        canonical_to_aliases[canonical].add(raw_entity)
        canonical_to_aliases[canonical].add(form_entity)
        alias_to_canonical[raw_entity] = canonical
        alias_to_canonical[form_entity] = canonical

    # 转换为JSON可序列化格式
    lexicon = {
        "canonical_to_aliases": {k: sorted(list(v)) for k, v in canonical_to_aliases.items()},
        "alias_to_canonical": alias_to_canonical
    }

    # ── 7. 保存 Lexicon ──
    lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
    logger.log_info(f"✅ Lexicon saved to {lexicon_path}")
    logger.log_info(f"   Canonical entities: {len(canonical_to_aliases)}")
    logger.log_info(f"   Total aliases: {len(alias_to_canonical)}\n")

    # ── 8. 应用归一化到所有经验 ──
    logger.log_info("Step 7: Applying normalization to all experiences...")
    output_records = []

    for line in lines:
        record = json.loads(line)
        core_entities = record.get("core_entities", [])

        # 收集canonical entities
        canonical_set = set()
        for ent_dict in core_entities:
            raw_entity = ent_dict.get("entity", "")
            if raw_entity:
                canonical = alias_to_canonical.get(raw_entity, normalize_form(raw_entity))
                canonical_set.add(canonical)

        record["canonical_entities"] = sorted(list(canonical_set))
        output_records.append(record)

    # ── 9. 原地更新文件 ──
    with open(input_path, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.log_info(f"✅ Updated {input_path} with canonical_entities field")
    logger.log_info("\n" + "="*80)
    logger.log_info("STEP 2.5 COMPLETE")
    logger.log_info("="*80 + "\n")


# ─────────────────────────── 入口 ───────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize core_entities in JSONL (Form + Semantic)"
    )
    parser.add_argument("input",
                        help="Path to core_entities.jsonl")
    parser.add_argument("--lexicon",
                        default="data/lexicon/lexicon.json",
                        help="Path to lexicon JSON (default: data/lexicon/lexicon.json)")
    parser.add_argument("--threshold",
                        type=float, default=SIMILARITY_THRESHOLD,
                        help=f"Similarity threshold for clustering (default: {SIMILARITY_THRESHOLD})")
    args = parser.parse_args()

    normalize_jsonl(args.input, args.lexicon, args.threshold)
