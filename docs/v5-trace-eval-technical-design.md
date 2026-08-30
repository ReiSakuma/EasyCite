# EasyCite V5 Trace / Eval 技术设计

V5 的目标是把 EasyCite 从“能推荐引用”推进到“知道自己为什么推荐、为什么没推荐、如何证明版本变好”。

本版本不改变核心推荐算法，不接入 LangGraph runtime，先补齐后续 StateGraph / LangGraph / Memory 需要依赖的可观测基础。

## 1. 新增能力

### 1.1 Run-level Trace

每次引用推荐都会生成 `run_id`，并记录 `trace_events`。

每个事件包含：

- `run_id`
- `stage`
- `message`
- `candidate_count_before`
- `candidate_count_after`
- `latency_ms`
- `error`
- `related_candidate_keys`

当前覆盖的阶段包括：

- `start`
- `claim_analysis`
- `query_planning`
- `current_bib_search`
- `zotero_retrieval`
- `forced_debug_probe`
- `title_doi_dedupe`
- `scope_filter`
- `metadata_rank`
- `evidence_shortlist`
- `evidence_verification`
- `evidence_rank`
- `final_recommendations`
- `stop`

### 1.2 Candidate Lifecycle Trace

每篇候选文献会携带 `trace` 字段，用于说明它在推荐流程中的生命周期：

- 从哪里被召回
- 是否通过 scope filter
- metadata rank 排名多少
- 是否进入 evidence shortlist
- LLM/evidence verdict 是什么
- 是否进入主推荐区
- 是否只保留在诊断候选区

### 1.3 UI 展示

候选面板新增：

- 每张候选卡片的 “为什么推荐/未推荐”
- 页面底部 “运行 Trace”

这让用户能直接看到每个阶段的候选数量变化、耗时和决策摘要。

### 1.4 Eval Harness

新增 `eval/` 目录：

```text
eval/
├── README.md
├── cases/
│   └── nerf_slam_claims.jsonl
└── run_eval.py
```

`run_eval.py` 会调用同一套后端推荐代码并输出：

- `citation_needed_accuracy`
- `target_recall_at_5`
- `diagnostic_recall_at_10`
- `forbidden_recommendation_rate`
- `unsupported_main_recommendations`
- `average_latency_ms`

## 2. 设计取舍

V5 暂时不接 LangGraph，因为 LangGraph 的最大收益来自 checkpoint、replay 和 human-in-the-loop，而这些能力需要先有稳定的 state 和 trace schema。

因此 V5 的策略是：

```text
先补 Trace / Eval
↓
再抽象 State / Node
↓
最后接 LangGraph runtime
```

## 3. 验证方式

```bash
cd <repository-root>
npm run compile
.venv/bin/python -m pytest src/test/python_tests
.venv/bin/python eval/run_eval.py --cases eval/cases/nerf_slam_claims.jsonl
```

