# EasyCite v6 Agent State 与 Run Record 技术设计

## 目标

v6 的目标不是继续增加新的检索按钮，而是根据面向 Agent 岗位面试的技术路线，把 v5 已经跑通的推荐流程整理成更可观测、可复盘、可评测的 Agent workflow。

核心新增能力：

- 每次引用推荐运行都会生成一份 LangGraph-compatible Agent State 快照。
- 快照会持久化到当前工作区 `.citation-agent/runs/{run_id}.json`。
- 候选审核面板新增 `Agent State` 区域，展示本次运行的关键状态、候选流转计数和 run record 打开入口。
- 为后续 v7 接入 checkpoint/replay、bad case analyzer、LangGraph 编排奠定数据结构基础。

## 新增状态结构

v6 新增的 run record 采用如下顶层结构：

```text
AgentRunRecord
├── schema_version
├── run_id
├── created_at
├── request
├── runtime
│   ├── model
│   └── sources
├── claim_profile
├── retrieval_plan
├── candidate_pool
├── verification
├── output
└── trace
```

这对应技术路线中的：

- `request`
- `claim_profile`
- `retrieval_plan`
- `candidate_pool`
- `verification`
- `output`
- `trace`

目前它还不是完整 LangGraph State，但字段边界已经按未来 StateGraph 迁移方式设计。

## 持久化位置

每次运行保存到当前工作区：

```text
.citation-agent/runs/{run_id}.json
.citation-agent/runs/latest.json
```

其中：

- `{run_id}.json` 是完整状态快照。
- `latest.json` 只保存最近一次 run id 和文件路径，后续可以用于“打开最近运行”“重放最近 bad case”。

## UI 改动

候选审核 Webview 新增 `Agent State` 区块：

- Run ID
- LLM provider / model / API key 是否可见
- query planner 来源：`llm` 或 `local`
- 候选流转计数：召回、去重、范围过滤、shortlist、验证、最终推荐、诊断候选
- retrieval plan reason
- 最终 citation keys
- 打开 Run Record JSON 按钮

这让用户可以在不看输出日志的情况下判断：

- 这次到底有没有调用 LLM。
- 候选在哪一步减少。
- 是否是 Zotero 召回少、metadata rank 问题、evidence shortlist 问题，还是 LLM verification 问题。

## 修改文件

- `bundled/tool/citation_agent/agent_state.py`
  - 新增 Agent State 快照构建与持久化。

- `bundled/tool/citation_agent/recommender.py`
  - 在推荐流程末尾构建 `agent_state`。
  - 保存 run record。
  - `RecommendationBundle` 返回 `agent_state` 和 `agent_run_path`。

- `bundled/tool/citation_agent/schemas.py`
  - `AuditClaimResult` 增加 `agent_state` 和 `agent_run_path`。

- `bundled/tool/lsp_server.py`
  - LSP 返回新增字段。

- `src/types/protocol.ts`
  - 前端协议增加 `AgentRunState`。

- `src/commands/auditSelection.ts`
  - 候选面板渲染 Agent State 摘要。
  - 支持打开当前 run record JSON。

- `package.json` / `package-lock.json` / `.vscode/launch.json`
  - 版本和调试名升级为 v6。

## 后续方向

v6 之后推荐继续推进：

1. Bad Case Analyzer
   - 从 run record 一键生成 eval JSONL case。
   - 标记“应该推荐但未推荐”的目标论文。

2. Replay
   - 固定前序候选池，只重跑 metadata rank / evidence verification。
   - 用于比较 prompt、模型和 ranking 策略。

3. LangGraph Phase 1
   - 将 recommender 中的顺序流程拆成 node 函数。
   - 先保持无 LangGraph 依赖，但每个 node 输入/输出都写入 state。

4. Tool Registry
   - Zotero、BBT、PDF、Embedding、LLM 都包装成统一 ToolResult。
   - 记录 tool latency、retry、error、fallback。

