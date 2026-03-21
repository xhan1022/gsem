"""
retriver.py
图遍历检索主模块。所有 prompt 工程（含 system prompt）集中在此文件。

入口函数：retrieve(case, finder)
  - 输入：case dict（必须含 description 字段）、FindStart 实例
  - 输出：被 agent 选中的经验 dict 列表（含 id 字段）
"""

import json
import os
import re
from collections import defaultdict
from typing import Optional

from agent import call_llm


# ── 加载图数据 ────────────────────────────────────────────────────────────────

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_GRAPH_PATH = os.path.join(_MODULE_DIR, "graph", "experience_graph.json")

with open(_GRAPH_PATH, "r", encoding="utf-8") as f:
    _graph_data = json.load(f)

# 节点信息字典：{ node_id: node_dict }
node_info: dict = {node["id"]: node for node in _graph_data["nodes"]}

# 邻接表：{ node_id: [(neighbor_id, edge_weight), ...] }
adj: dict = defaultdict(list)
for _link in _graph_data["links"]:
    _src = _link["source"]
    _tgt = _link["target"]
    _w = float(_link.get("weight", _link.get("W", 0.0)))
    adj[_src].append((_tgt, _w))
    adj[_tgt].append((_src, _w))  # 无向图，双向添加


# ── 图工具函数 ────────────────────────────────────────────────────────────────

def get_top_neighbors(
    node_id: str,
    visited: set[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    获取指定节点 top-k 未访问邻居，按综合分数降序排列。
    综合分数 = (边权重 + 邻居节点 quality) / 2
    """
    candidates = []
    for neighbor_id, edge_w in adj.get(node_id, []):
        if neighbor_id in visited:
            continue
        if neighbor_id not in node_info:
            continue
        quality = float(node_info[neighbor_id].get("quality", 0.0))
        score = (edge_w + quality) / 2.0
        candidates.append((neighbor_id, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def _filter_unvisited(
    layer: list[tuple[str, float]],
    visited: set[str],
) -> list[tuple[str, float]]:
    """从层候选列表中过滤已访问节点。"""
    return [(nid, s) for nid, s in layer if nid not in visited]


def _get_node_context(node_id: str) -> str:
    """返回节点 context；若无 context，则回退到 content。"""
    node = node_info.get(node_id, {})
    return str(node.get("context") or node.get("content") or "N/A")


def _get_node_condition(node_id: str) -> str:
    """返回节点 condition。"""
    return str(node_info.get(node_id, {}).get("condition", "N/A"))


def _format_candidates(candidates: list[tuple[str, float]]) -> str:
    """
    将候选节点列表格式化为 prompt 文本段。
    按你的要求，仅提供 id 和 condition。
    """
    if not candidates:
        return ""
    lines = []
    for nid, _ in candidates:
        lines.append(f"- {nid} | condition: {_get_node_condition(nid)}")
    return "\n".join(lines)


def _find_backtrack_layer(
    layer_stack: list[list[tuple[str, float]]],
    visited: set[str],
) -> tuple[Optional[int], list[tuple[str, float]]]:
    """
    从最近的祖先层开始，向上寻找第一个仍有未访问节点的层。
    返回：(层索引, 未访问候选列表)
    若不存在可回溯层，则返回 (None, [])
    """
    if len(layer_stack) <= 1:
        return None, []

    # 当前层的 next-layer 在 layer_stack[-1]
    # 回溯层应从倒数第 2 层开始向上找
    for idx in range(len(layer_stack) - 2, -1, -1):
        candidates = _filter_unvisited(layer_stack[idx], visited)
        if candidates:
            return idx, candidates
    return None, []


def _fallback_followup_action(
    next_layer: list[tuple[str, float]],
    backtrack_candidates: list[tuple[str, float]],
) -> tuple[int, str]:
    """
    当 LLM 未给出合法的非 a1 后续动作时，自动兜底：
    优先 a2，其次 a3，否则 a4
    """
    if next_layer:
        return 2, next_layer[0][0]
    if backtrack_candidates:
        return 3, backtrack_candidates[0][0]
    return 4, "none"


# ══════════════════════════════════════════════════════════════════════════════
# Prompt 工程（system prompt 与各步 user prompt 统一管理）
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a clinical experience retrieval agent traversing a graph of reusable clinical reasoning experiences.

Your PRIMARY GOAL: collect experiences (a1) that are relevant to the clinical query. You must collect at least 1 experience before stopping.

At each step, you will receive:
1. the clinical query
2. the current node's condition and context
3. optional forward candidates for action a2
4. optional backtrack candidates for action a3
5. the actions allowed in the current step

=== Actions ===
a1  COLLECT   — select the current node as a relevant experience
a2  EXPLORE   — move to one node from the forward candidates
a3  BACKTRACK — move to one node from the backtrack candidates
a4  STOP      — stop the retrieval process

=== Decision guide (follow this order every step) ===
1. Read the current node's condition and context carefully.
2. Compare it with the clinical query.
3. If it is relevant or partially relevant to the query → use a1 to collect it, then choose a2/a3/a4.
4. If it is clearly irrelevant → skip a1, just use a2, a3, or a4 to move on.
5. Only use a4 (STOP) when you have collected enough relevant experiences or all candidates are exhausted.

=== Rules ===
- Only choose from the actions explicitly listed as allowed this step.
- If you choose a1, you SHOULD also output one more action after it (a2, a3, or a4).
  If you omit the second action after a1, the system will handle it automatically.
- Do not use a1 again for a node that has already been collected.

=== Output format ===
Thought: <one line comparing current node with query, then your decision>
<action>NUMBER,node_id</action>
<action>NUMBER,node_id</action>   (recommended when the first action is a1)

Return only the thought line and the action tag(s).
"""


def _build_step_prompt(
    query: str,
    current_id: str,
    next_layer: list[tuple[str, float]],
    backtrack_candidates: list[tuple[str, float]],
    step: int,
    selected_ids: list[str],
) -> str:
    """
    每步 user prompt：
    - 当前节点 condition / context
    - 可探索节点 id / condition（若存在）
    - 可回溯节点 id / condition（若存在）
    - 本轮允许动作
    """
    current_condition = _get_node_condition(current_id)
    current_context = _get_node_context(current_id)
    selected_set = set(selected_ids)

    allowed_actions = []
    if current_id not in selected_set:
        allowed_actions.append("a1")
    if next_layer:
        allowed_actions.append("a2")
    if backtrack_candidates:
        allowed_actions.append("a3")
    allowed_actions.append("a4")

    lines = [
        f"=== Step {step} ===",
    ]

    if query:
        if step == 0:
            lines.extend([
                "=== Clinical Query ===",
                query,
                "",
            ])
        else:
            short_query = query[:300] + "..." if len(query) > 300 else query
            lines.extend([
                "=== Clinical Query (reminder) ===",
                short_query,
                "",
            ])

    lines.extend([
        "=== Current node ===",
        f"id: {current_id}",
        f"condition: {current_condition}",
        f"context: {current_context}",
        "",
        f"Collected so far: [{', '.join(selected_ids) if selected_ids else 'none'}]",
        "",
        f"Allowed actions this step: {', '.join(allowed_actions)}",
    ])

    if current_id not in selected_set:
        lines.append(
            "If you choose a1, you should also output one more action after a1 (recommended)."
        )

    if next_layer:
        lines.extend([
            "",
            "=== Forward candidates for a2 ===",
            _format_candidates(next_layer),
        ])

    if backtrack_candidates:
        lines.extend([
            "",
            "=== Backtrack candidates for a3 ===",
            _format_candidates(backtrack_candidates),
        ])

    lines.extend([
        "",
        "Action reminder:",
    ])

    if current_id not in selected_set:
        lines.append("- a1: collect the current node; if used, you must also output a second action")
    if next_layer:
        lines.append("- a2: choose one node from the forward candidates")
    if backtrack_candidates:
        lines.append("- a3: choose one node from the backtrack candidates")
    lines.append("- a4: stop retrieval")

    lines.extend([
        "",
        "Output format:",
        "Thought: ...",
        "<action>NUMBER,node_id</action>",
        "<action>NUMBER,node_id</action>   (only if the first action is a1)",
    ])

    return "\n".join(lines)


# ── Action 解析 ───────────────────────────────────────────────────────────────

def parse_actions(response: str) -> list[tuple[int, str]]:
    """
    从 agent 响应中解析一个或多个 <action>num,id</action>。
    同时支持数字格式和字母前缀格式：
      <action>1,exp_0001</action>
      <action>a1,exp_0001</action>
    a4 的 node_id 可省略：
      <action>4</action>
      <action>a4</action>
    """
    if not response:
        return []
    matches = re.findall(
        r"<action>\s*a?(\d+)\s*(?:,\s*([^<\s]+))?\s*</action>",
        response,
    )
    return [(int(num), (nid.strip() if nid else "none")) for num, nid in matches]


# ── 核心遍历逻辑 ──────────────────────────────────────────────────────────────

def _record_selection(current_id: str, selected: list[dict]) -> bool:
    """将当前节点的完整经验内容加入 selected 列表（去重）。"""
    if any(e["id"] == current_id for e in selected):
        print(f"    ℹ 已选取过: {current_id}")
        return False

    exp = {"id": current_id, **node_info.get(current_id, {})}
    selected.append(exp)

    print(f"    ✔ 选取经验: {current_id}")
    for k, v in exp.items():
        if k != "id":
            print(f"       {k}: {v}")
    return True


def retrieve(case: dict, finder, max_steps: int = 60) -> list[dict]:
    """
    入口函数：使用 agent 遍历经验图，返回与 case 相关的经验列表。

    参数：
        case:      查询 case dict，必须含 "description" 字段。
        finder:    已初始化的 FindStart 实例。
        max_steps: 最大遍历步数。

    返回：
        被 agent 选中的经验 dict 列表（含 id 字段），按选取顺序排列。

    逻辑说明：
        1. finder.find(case) 返回候选起始点列表
        2. 取第一个作为当前入口
        3. 其余起始点作为第一步可回溯节点
        4. 当前节点的 top-5 邻居作为当前可探索节点
        5. agent 输出:
              <action>1,node_i</action><action>2|3|4,node_j</action>
           或 <action>2,node_i</action>
           或 <action>3,node_i</action>
           或 <action>4,none</action>
        6. 当无可探索 / 无可回溯 / 已选满 5 条 / agent 选择 a4 时结束
    """
    query = case.get("description", "")

    # ── 1. 获取起始节点列表 ───────────────────────────────────────────────────
    start_ids: list[str] = finder.find(case)
    if not start_ids:
        raise ValueError("finder.find() 返回空列表，无法开始遍历")

    # 只保留图中存在的起始节点
    start_ids = [nid for nid in start_ids if nid in node_info]
    if not start_ids:
        raise ValueError("finder.find() 返回的节点均不在图中，无法开始遍历")

    current_id: str = start_ids[0]
    visited: set[str] = {current_id}
    selected: list[dict] = []

    # ── 2. 初始化层级栈 ───────────────────────────────────────────────────────
    # 将其余起始点与当前起始点的邻居合并为同一候选池（layer_stack[0]）。
    # 这样 agent 回溯时随时可跳转到任意高质量起始点，
    # 避免因图深度过大导致其他起始点永远无法被访问。
    other_start_candidates: list[tuple[str, float]] = [
        (nid, float(node_info[nid].get("quality", 0.0)))
        for nid in start_ids[1:]
        if nid in node_info
    ]

    start_neighbors: list[tuple[str, float]] = get_top_neighbors(current_id, visited, top_k=5)

    # 合并并按分数降序，作为初始候选池（第 0 层）
    combined_pool: list[tuple[str, float]] = sorted(
        other_start_candidates + start_neighbors,
        key=lambda x: x[1],
        reverse=True,
    )
    layer_stack: list[list[tuple[str, float]]] = [combined_pool]

    print(f"\n{'=' * 60}")
    print(f"[Retrieve] Query  : {query[:120]}...")
    print(f"[Retrieve] Start  : {current_id}")
    print(f"[Retrieve] Pool   : {[nid for nid, _ in combined_pool[:8]]}")
    print(f"{'=' * 60}")

    # ── 3. 初始化多轮对话 ────────────────────────────────────────────────────
    conversation: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
    ]

    # ── 4. 主遍历循环 ─────────────────────────────────────────────────────────
    for step in range(max_steps):
        selected_ids = [e["id"] for e in selected]
        can_collect = current_id not in selected_ids

        next_layer = _filter_unvisited(layer_stack[-1], visited)
        backtrack_layer_idx, backtrack_candidates = _find_backtrack_layer(layer_stack, visited)

        # 结束条件：
        # 1) 已选满 5 条
        # 2) 当前节点已无法 a1 且无 a2 / a3
        if len(selected) >= 5:
            print(f"[Step {step}] ■ 已选满 5 条经验，结束遍历")
            break

        if not can_collect and not next_layer and not backtrack_candidates:
            print(f"[Step {step}] ■ 无可探索/回溯节点，且当前节点已处理，结束遍历")
            break

        prompt = _build_step_prompt(
            query=query,
            current_id=current_id,
            next_layer=next_layer,
            backtrack_candidates=backtrack_candidates,
            step=step,
            selected_ids=selected_ids,
        )
        conversation.append({"role": "user", "content": prompt})

        response = call_llm(conversation)
        conversation.append({"role": "assistant", "content": response})

        lines = response.strip().splitlines() if response else []
        last_line = lines[-1] if lines else "<empty response>"
        print(f"[Step {step}] {last_line}")

        actions = parse_actions(response)
        if not actions:
            print(f"[Step {step}] ⚠ 无法解析 action，原始响应: {response!r}")
            break

        # 先处理第一动作；如果第一动作是 a1，则必须再接第二动作
        first_action, first_target = actions[0]
        pending_action: tuple[int, str]

        # ── a1：选取当前节点经验（并要求继续一个后续动作）─────────────────
        if first_action == 1:
            if not can_collect:
                print(f"[Step {step}] a1 ⚠ 当前节点已选中过，忽略 a1")
            else:
                _record_selection(current_id, selected)
                selected_ids = [e["id"] for e in selected]

                if len(selected) >= 5:
                    print(f"[Step {step}] ■ 已选满 5 条经验，结束遍历")
                    break

            if len(actions) >= 2:
                pending_action = actions[1]
            else:
                fallback = _fallback_followup_action(next_layer, backtrack_candidates)
                print(f"[Step {step}] a1 ⚠ 未提供第二动作，fallback → {fallback}")
                pending_action = fallback

        else:
            pending_action = actions[0]

        action, target_id = pending_action

        # ── a2：向前探索一层 ────────────────────────────────────────────────
        if action == 2:
            if not next_layer:
                print(f"[Step {step}] a2 ✗ 当前无可探索节点")
                fallback = _fallback_followup_action([], backtrack_candidates)
                if fallback[0] == 3:
                    action, target_id = fallback
                elif fallback[0] == 4:
                    print(f"[Step {step}] a2 → fallback a4，结束遍历")
                    break
                else:
                    continue

            if action == 2:
                valid_ids = [nid for nid, _ in next_layer]
                if target_id not in valid_ids:
                    print(f"[Step {step}] a2 ⚠ {target_id} 不在探索候选中，fallback → {valid_ids[0]}")
                    target_id = valid_ids[0]

                visited.add(target_id)

                # 当前 next_layer 成为新节点的“上一层”，新节点的 next_layer 需要压栈
                child_neighbors = get_top_neighbors(target_id, visited, top_k=5)
                layer_stack.append(child_neighbors)
                current_id = target_id

                print(f"[Step {step}] a2 ↓ 移动到: {current_id}（深度 {len(layer_stack) - 1}）")
                continue

        # ── a3：向后回溯 ────────────────────────────────────────────────────
        if action == 3:
            # 每轮实时重算，避免 a1 后状态变动造成不一致
            backtrack_layer_idx, backtrack_candidates = _find_backtrack_layer(layer_stack, visited)

            if backtrack_layer_idx is None or not backtrack_candidates:
                print(f"[Step {step}] a3 ✗ 当前无可回溯节点")
                fallback = _fallback_followup_action(next_layer, [])
                if fallback[0] == 2:
                    action, target_id = fallback
                elif fallback[0] == 4:
                    print(f"[Step {step}] a3 → fallback a4，结束遍历")
                    break
                else:
                    continue

            if action == 3:
                valid_ids = [nid for nid, _ in backtrack_candidates]
                if target_id not in valid_ids:
                    print(f"[Step {step}] a3 ⚠ {target_id} 不在回溯候选中，fallback → {valid_ids[0]}")
                    target_id = valid_ids[0]

                visited.add(target_id)

                # 回溯到找到的祖先层，并从该层所选目标继续展开其下一层
                layer_stack = layer_stack[: backtrack_layer_idx + 1]
                target_neighbors = get_top_neighbors(target_id, visited, top_k=5)
                layer_stack.append(target_neighbors)
                current_id = target_id

                print(f"[Step {step}] a3 ↑ 回溯到: {current_id}（深度 {len(layer_stack) - 1}）")
                continue

        # ── a4：结束遍历 ────────────────────────────────────────────────────
        if action == 4:
            print(f"[Step {step}] a4 ■ Agent 结束遍历")
            break

        # ── 未知动作 ────────────────────────────────────────────────────────
        print(f"[Step {step}] ⚠ 未知 action: {action}")
        break

    print(f"\n[Done] 共选取 {len(selected)} 条经验: {[e['id'] for e in selected]}")

    # ── 5. 兜底：GSEM 未选中任何经验时，注入初始点经验 ───────────────────────
    if not selected:
        print("[Done] GSEM 未召回任何经验，注入初始点经验作为兜底")
        for nid in start_ids:
            if len(selected) >= 5:
                break
            _record_selection(nid, selected)
        print(f"[Done] 兜底注入 {len(selected)} 条经验: {[e['id'] for e in selected]}")

    return selected


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    _this_dir = str(Path(__file__).resolve().parent)
    for p in [_this_dir, str(_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from find_start_two_stage import FindStartTwoStage

    # ── 加载 MedRBench 测试数据 ───────────────────────────────────────────────
    MEDRB_PATH = str(_root / "evaluation/medrb/data/splits/diagnosis/test.json")
    N_CASES = 10  # 测试 case 数量

    def _load_medrb_cases(path: str, n: int) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cases = []
        for cid, raw in data.items():
            gc = raw.get("generate_case", {})
            description = gc.get("case_summary", "").strip()
            if not description:
                continue
            cases.append({
                "case_id":     cid,
                "description": description,
                "answer":      gc.get("final_diagnosis", ""),
            })
            if len(cases) >= n:
                break
        return cases

    cases = _load_medrb_cases(MEDRB_PATH, N_CASES)
    print(f"已加载 {len(cases)} 个 MedRBench 测试 case\n")

    finder = FindStartTwoStage()

    # ── 逐 case 检索并输出相关性对比 ─────────────────────────────────────────
    SEP = "=" * 70
    for i, case in enumerate(cases, 1):
        print(f"\n{SEP}")
        print(f"[Case {i}/{len(cases)}] {case['case_id']}")
        print(f"Ground truth: {case['answer']}")
        print(f"Description (前200字): {case['description'][:200]}...")
        print(SEP)

        selected = retrieve(case, finder)

        print(f"\n── 检索结果（共 {len(selected)} 条）──")
        if not selected:
            print("  ⚠ 未召回任何经验")
        for rank, exp in enumerate(selected, 1):
            print(f"  [{rank}] {exp['id']}")
            print(f"      condition : {exp.get('condition', 'N/A')}")
            print(f"      content   : {str(exp.get('content', 'N/A'))[:200]}")
        print()
