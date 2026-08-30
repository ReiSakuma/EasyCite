# EasyCite v8 LangGraph-Compatible Node Refactor

## 目标

v8 的目标不是立刻引入 LangGraph 依赖，而是先把现有稳定的推荐流程整理成可观测、可持久化、可迁移的 graph 形态：

- 每次运行仍由现有 Python recommender 执行，避免一次性重写带来回归。
- Run Record 中新增 `graph` 字段，记录节点、边、checkpoint 策略。
- VS Code 候选审核面板的 `Agent State` 区域展示 graph 摘要。
- 后续 v9 可以把 `linear-node-runner` 替换为真实 LangGraph `StateGraph`。

## 设计

v8 新增 `bundled/tool/citation_agent/agent_nodes/`：

- `state.py`
  - 定义 `EasyCiteState`，作为未来 LangGraph state 的 TypedDict 轮廓。
  - 定义 `AgentNodeRecord`，用于把 trace event 转成节点记录。
- `runner.py`
  - 定义 `STAGE_TO_NODE`，把现有 trace stage 映射到语义化 graph node。
  - 定义 `GRAPH_EDGES`，描述推荐流程中的节点连接关系。
  - 定义 `CHECKPOINT_NODES`，标记后续最值得恢复和复现的节点。
  - 提供 `graph_from_trace_events()`，从一次运行的 trace 生成 graph snapshot。

当前节点包括：

- `configure_runtime`
- `analyze_claim`
- `plan_retrieval`
- `recall_current_bib`
- `recall_zotero`
- `debug_target_probe`
- `dedupe_candidates`
- `scope_filter`
- `metadata_rank`
- `select_evidence_targets`
- `verify_evidence`
- `rank_verified_evidence`
- `calibrate_recommendations`
- `prepare_review_payload`

## Run Record 变化

`bundled/tool/citation_agent/agent_state.py` schema 从：

```text
v7.agent_state.1
```

升级为：

```text
v8.agent_state.1
```

并新增：

```json
{
  "graph": {
    "schema_version": "v8.graph.1",
    "framework": "langgraph-compatible",
    "runtime": "linear-node-runner",
    "nodes": [],
    "edges": [],
    "checkpoint_policy": {
      "checkpoint_after": [],
      "storage": ".citation-agent/runs/{run_id}.json",
      "replay_status": "trace_replay_ready"
    }
  }
}
```

这意味着一次推荐不仅能看到结果，还能看到结果是如何从各阶段流转出来的。

## 用户端变化

普通用户的候选推荐、插入引用、BibTeX 写入、Zotero 范围限定等使用方式不变。

候选审核页 `Agent State` 区域新增 `Graph` 摘要，例如：

```text
15 nodes · 18 edges · 7 checkpoints · linear-node-runner
```

这让开发者和高级用户可以快速判断本次运行是否完整经过了 claim 分析、检索规划、证据验证、最终推荐等节点。

## 为什么不直接上 LangGraph

目前 EasyCite 的难点不只是“流程编排”，还包括：

- VS Code 前端与 Python 后端的 LSP 通信稳定性。
- Zotero / Better BibTeX / BibTeX 文件写入。
- 本地 embedding、PDF evidence 检索和 LLM evidence judging 的性能折中。
- 用户端候选审核 UI 的可用性。

因此 v8 采用保守的渐进式迁移：

1. 先把现有流程映射成 graph。
2. 持久化 graph state 和 checkpoint 策略。
3. 让 UI 和测试开始依赖稳定的 graph schema。
4. 后续再把内部执行器替换成 LangGraph。

## 后续 v9 建议

v9 可以在 v8 的基础上继续推进：

- 引入可选 LangGraph 依赖。
- 用 `StateGraph` 显式定义节点函数和 conditional edges。
- 为 Zotero 召回、evidence 验证、LLM judging 增加可恢复 checkpoint。
- 增加基于 run record 的 replay 命令，用同一输入复现一次历史运行。
- 将 eval bad case 与 graph replay 合并，形成自动回归评测。
