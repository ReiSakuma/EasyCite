# EasyCite v10: Eval Harness

## 目标

v10 的目标是把人工发现的 bad case 固化成可重复运行的评测用例，让每次策略、prompt、检索或证据验证改动都能用指标判断是否退化。

这不是用户侧的新按钮，而是 Agent 工程闭环：

```text
bad case -> eval case -> run eval -> metrics report -> regression guard
```

## 新增模块

- `bundled/tool/citation_agent/eval_harness.py`
  - 负责加载 JSONL case、运行 EasyCite、计算指标、生成报告。
- `eval/run_eval.py`
  - 命令行入口。
- `eval/cases/nerf_slam_claims.jsonl`
  - 初始 regression cases。
- `eval/reports/`
  - 运行后生成 JSON / Markdown 报告。

## Eval Case

每条 case 是一行 JSON：

```json
{
  "id": "nerf_slam_001",
  "claim": "Recent neural and differentiable representations...",
  "context": "For many years, dense SLAM systems...",
  "citation_needed": true,
  "gold_papers": ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"],
  "acceptable_papers": ["iMAP", "NICE-SLAM", "Co-SLAM", "CP-SLAM"],
  "forbidden_papers": ["pure geometric SLAM paper without neural/differentiable representation"],
  "expected_evidence_keywords": ["neural radiance field", "differentiable rendering"]
}
```

## 运行方式

```bash
cd <repository-root>
.venv/bin/python eval/run_eval.py \
  --cases eval/cases/nerf_slam_claims.jsonl \
  --output eval/reports/latest.json \
  --markdown-output eval/reports/latest.md
```

也可以通过 npm script：

```bash
npm run eval
```

## 关键指标

- `citation_needed_accuracy`
- `citation_needed_precision`
- `citation_needed_recall`
- `citation_needed_f1`
- `target_recall_at_5`
- `diagnostic_recall_at_10`
- `forbidden_recommendation_rate`
- `unsupported_recommendation_rate`
- `end_to_end_success_rate`
- `average_evidence_keyword_hit_rate`
- `average_latency_ms`
- `llm_timeout_rate`
- `tool_failure_rate`

其中 MVP 最重要的是：

```text
target_recall_at_5
diagnostic_recall_at_10
unsupported_recommendation_rate
forbidden_recommendation_rate
```

## Regression Guard

CLI 支持阈值：

```bash
.venv/bin/python eval/run_eval.py \
  --min-target-recall-at-5 0.7 \
  --max-unsupported-rate 0.2 \
  --max-forbidden-rate 0.05
```

如果指标不达标，命令返回 exit code `2`。

## 与 Trace / LangGraph 的关系

v10 eval 不重新实现推荐逻辑，而是直接调用现有后端：

```text
Eval Harness
  -> CitationNeedDetector
  -> CitationRecommender
  -> LangGraph StateGraph runtime
  -> Trace / Agent State / Target Trace
```

当 case 配置了 `gold_papers` / `acceptable_papers` 时，harness 会把这些目标文献临时注入 `debugTargetPapers`，这样报告中可以保存目标文献生命周期，辅助判断漏召回到底发生在哪一环。

## 面试表达

v10 可以这样讲：

> 我没有只靠手动试几句话判断推荐效果，而是把用户真实发现的 bad case 固化成 JSONL eval case。每次修改 query planner、metadata rank、evidence verification 或 prompt 后都会跑 eval，观察 target_recall@5、unsupported recommendation rate 和 latency。这样可以证明优化有效，也能防止旧 bad case 回归。
