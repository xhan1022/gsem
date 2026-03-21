"""
retriver_multi_start.py
多起点图遍历检索模块。

入口函数：retrieve(case, finder)
  - 输入：case dict（必须含 description 字段）、FindStart 实例
  - 输出：对每个初始点分别遍历后合并得到的经验列表（含 id 字段）

与 retriver.py 的区别：
  1. finder.find(case) 返回多个初始点后，不再只从第一个点开始。
  2. 对每个初始点分别执行一次图遍历。
  3. 若某个初始点这一路没有选中任何经验，则直接将该初始点加入结果。
  4. 最终将所有起点得到的结果按遍历顺序去重合并，提升整体召回。
"""

from typing import Optional

from retriver import (
    _SYSTEM_PROMPT,
    _build_step_prompt,
    _fallback_followup_action,
    _filter_unvisited,
    _find_backtrack_layer,
    _record_selection,
    call_llm,
    get_top_neighbors,
    node_info,
    parse_actions,
)


def _retrieve_from_start(
    query: str,
    start_id: str,
    max_steps: int,
) -> list[dict]:
    """Run one independent traversal from a single start node."""
    current_id = start_id
    visited: set[str] = {current_id}
    selected: list[dict] = []

    start_neighbors = get_top_neighbors(current_id, visited, top_k=5)
    layer_stack: list[list[tuple[str, float]]] = [start_neighbors]

    print(f"\n{'=' * 60}")
    print(f"[Retrieve:SingleStart] Query  : {query[:120]}...")
    print(f"[Retrieve:SingleStart] Start  : {current_id}")
    print(f"[Retrieve:SingleStart] Pool   : {[nid for nid, _ in start_neighbors[:8]]}")
    print(f"{'=' * 60}")

    conversation: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
    ]

    for step in range(max_steps):
        selected_ids = [e["id"] for e in selected]
        can_collect = current_id not in selected_ids

        next_layer = _filter_unvisited(layer_stack[-1], visited)
        backtrack_layer_idx, backtrack_candidates = _find_backtrack_layer(layer_stack, visited)

        if len(selected) >= 5:
            print(f"[Start {start_id} | Step {step}] ■ 已选满 5 条经验，结束遍历")
            break

        if not can_collect and not next_layer and not backtrack_candidates:
            print(f"[Start {start_id} | Step {step}] ■ 无可探索/回溯节点，且当前节点已处理，结束遍历")
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
        print(f"[Start {start_id} | Step {step}] {last_line}")

        actions = parse_actions(response)
        if not actions:
            print(f"[Start {start_id} | Step {step}] ⚠ 无法解析 action，原始响应: {response!r}")
            break

        first_action, _ = actions[0]
        pending_action: tuple[int, str]

        if first_action == 1:
            if not can_collect:
                print(f"[Start {start_id} | Step {step}] a1 ⚠ 当前节点已选中过，忽略 a1")
            else:
                _record_selection(current_id, selected)
                selected_ids = [e["id"] for e in selected]

                if len(selected) >= 5:
                    print(f"[Start {start_id} | Step {step}] ■ 已选满 5 条经验，结束遍历")
                    break

            if len(actions) >= 2:
                pending_action = actions[1]
            else:
                fallback = _fallback_followup_action(next_layer, backtrack_candidates)
                print(f"[Start {start_id} | Step {step}] a1 ⚠ 未提供第二动作，fallback → {fallback}")
                pending_action = fallback
        else:
            pending_action = actions[0]

        action, target_id = pending_action

        if action == 2:
            if not next_layer:
                print(f"[Start {start_id} | Step {step}] a2 ✗ 当前无可探索节点")
                fallback = _fallback_followup_action([], backtrack_candidates)
                if fallback[0] == 3:
                    action, target_id = fallback
                elif fallback[0] == 4:
                    print(f"[Start {start_id} | Step {step}] a2 → fallback a4，结束遍历")
                    break
                else:
                    continue

            if action == 2:
                valid_ids = [nid for nid, _ in next_layer]
                if target_id not in valid_ids:
                    print(f"[Start {start_id} | Step {step}] a2 ⚠ {target_id} 不在探索候选中，fallback → {valid_ids[0]}")
                    target_id = valid_ids[0]

                visited.add(target_id)
                child_neighbors = get_top_neighbors(target_id, visited, top_k=5)
                layer_stack.append(child_neighbors)
                current_id = target_id

                print(f"[Start {start_id} | Step {step}] a2 ↓ 移动到: {current_id}（深度 {len(layer_stack) - 1}）")
                continue

        if action == 3:
            backtrack_layer_idx, backtrack_candidates = _find_backtrack_layer(layer_stack, visited)

            if backtrack_layer_idx is None or not backtrack_candidates:
                print(f"[Start {start_id} | Step {step}] a3 ✗ 当前无可回溯节点")
                fallback = _fallback_followup_action(next_layer, [])
                if fallback[0] == 2:
                    action, target_id = fallback
                elif fallback[0] == 4:
                    print(f"[Start {start_id} | Step {step}] a3 → fallback a4，结束遍历")
                    break
                else:
                    continue

            if action == 3:
                valid_ids = [nid for nid, _ in backtrack_candidates]
                if target_id not in valid_ids:
                    print(f"[Start {start_id} | Step {step}] a3 ⚠ {target_id} 不在回溯候选中，fallback → {valid_ids[0]}")
                    target_id = valid_ids[0]

                visited.add(target_id)
                layer_stack = layer_stack[: backtrack_layer_idx + 1]
                target_neighbors = get_top_neighbors(target_id, visited, top_k=5)
                layer_stack.append(target_neighbors)
                current_id = target_id

                print(f"[Start {start_id} | Step {step}] a3 ↑ 回溯到: {current_id}（深度 {len(layer_stack) - 1}）")
                continue

        if action == 4:
            print(f"[Start {start_id} | Step {step}] a4 ■ Agent 结束遍历")
            break

        print(f"[Start {start_id} | Step {step}] ⚠ 未知 action: {action}")
        break

    if not selected:
        print(f"[Start {start_id}] 未选中任何经验，回填起始点")
        _record_selection(start_id, selected)

    print(f"[Start {start_id}] 完成，共得到 {len(selected)} 条经验: {[e['id'] for e in selected]}")
    return selected


def retrieve(case: dict, finder, max_steps: int = 60) -> list[dict]:
    """
    多起点入口函数：对每个初始点分别执行独立遍历并合并结果。

    参数：
        case:      查询 case dict，必须含 "description" 字段。
        finder:    已初始化的 FindStart 实例。
        max_steps: 每个起始点的最大遍历步数。

    返回：
        所有起始点遍历结果合并后的经验 dict 列表（去重，按出现顺序保留）。
    """
    query = case.get("description", "")

    start_ids: list[str] = finder.find(case)
    if not start_ids:
        raise ValueError("finder.find() 返回空列表，无法开始遍历")

    start_ids = [nid for nid in start_ids if nid in node_info]
    if not start_ids:
        raise ValueError("finder.find() 返回的节点均不在图中，无法开始遍历")

    merged: list[dict] = []
    merged_ids: set[str] = set()

    print(f"\n{'#' * 60}")
    print(f"[Retrieve:MultiStart] 共 {len(start_ids)} 个起始点: {start_ids}")
    print(f"{'#' * 60}")

    for idx, start_id in enumerate(start_ids, 1):
        print(f"\n[Retrieve:MultiStart] 处理起始点 {idx}/{len(start_ids)}: {start_id}")
        per_start = _retrieve_from_start(query=query, start_id=start_id, max_steps=max_steps)
        for exp in per_start:
            exp_id = exp["id"]
            if exp_id in merged_ids:
                continue
            merged.append(exp)
            merged_ids.add(exp_id)

    print(f"\n[Retrieve:MultiStart] 最终合并 {len(merged)} 条经验: {[e['id'] for e in merged]}")
    return merged
