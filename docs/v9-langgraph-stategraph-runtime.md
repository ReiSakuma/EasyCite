# EasyCite v9: LangGraph StateGraph Runtime

## 目标

v8 已经把推荐流程整理成 LangGraph-compatible 的节点快照，但实际执行仍是单个线性 Python 函数。v9 的目标是把运行时真正迁移到 LangGraph：推荐流程由 `StateGraph` 编排，每个阶段都是显式节点，运行后仍保存 trace、agent state、候选诊断和 eval 数据。

## 架构变化

- 默认后端运行时变为 `citation-agent.agentRuntime=langgraph`。
- 新增 `bundled/tool/citation_agent/agent_graph/runner.py`，使用 `langgraph.graph.StateGraph` 构建推荐工作流。
- 保留 `agentRuntime=classic` 作为降级模式。若用户环境缺少 `langgraph`，后端会记录日志并自动回退到 v8 线性流程。
- 首次启动依赖检查把 `langgraph>=0.2,<1` 作为必需依赖，安装到 VS Code 扩展私有目录，不污染用户 base 环境。

## LangGraph 节点

v9 StateGraph 按如下节点顺序执行：

1. `initialize`: 构造 `TraceRecorder`、读取 LLM/来源配置、创建运行快照。
2. `validate_request`: 检查 claim 与最大推荐数量。
3. `analyze_claim`: 构建 claim profile。
4. `plan_retrieval`: 调用 LLM 或本地 planner 生成检索计划。
5. `recall_current_bib`: 从当前 TeX 工程 `.bib` 召回候选。
6. `recall_zotero`: 按 Zotero collection scope 召回候选，并支持 debug target probe。
7. `dedupe_candidates`: 按 DOI、标题、citation key 去重。
8. `scope_filter`: 用 claim scope 移除明显偏题候选。
9. `metadata_rank`: 对召回候选进行快速排序，但不作为唯一硬筛。
10. `select_evidence_targets`: 选择证据验证 shortlist，并保留诊断候选。
11. `verify_evidence`: 检索摘要/PDF passage，调用 LLM 或本地规则判断证据。
12. `rank_verified_evidence`: 按 verdict、置信度、来源稳定排序。
13. `finalize_recommendations`: 生成推荐区与未入选诊断区。
14. `persist_run`: 保存 run record，并标记 `framework=langgraph`、`runtime=langgraph-stategraph`。

## 用户端影响

v9 不改变主候选面板、灯泡菜单、Zotero 范围配置、BibTeX 写入等用户交互。用户能看到的主要变化是在开发者模式下打开运行状态时，run record 会显示真实 LangGraph runtime，而不是 v8 的 compatible 快照。

## 设置项

```json
{
  "citation-agent.agentRuntime": "langgraph"
}
```

可选值：

- `langgraph`: 默认值，使用真实 LangGraph StateGraph。
- `classic`: 使用 v8 线性流程，适合排查依赖问题或对比回归。

## 验证标准

- `npm run compile` 通过。
- `.venv/bin/python -m pytest src/test/python_tests` 通过。
- run record 中满足：
  - `schema_version = v9.agent_state.1`
  - `graph.schema_version = v9.graph.1`
  - `graph.framework = langgraph`
  - `graph.runtime = langgraph-stategraph`
  - trace 中包含 `graph_runtime` 阶段。
