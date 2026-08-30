# EasyCite v7 Bad Case Analyzer 技术设计

## 目标

v7 基于 v6 的 Agent State / Run Record，补上面向 Agent 工程闭环的下一步：把一次用户发现的问题导出成可回归评测样本。

这版新增的是最小可用 Bad Case Analyzer。它是开发者/研究者功能，默认不出现在普通用户界面中：

- 新增配置 `citation-agent.developerMode`，默认 `false`。
- 只有开启 developer mode 后，候选审核面板才会显示 `导出 Eval Bad Case`。
- 点击后会在当前候选审核界面的 `Agent State` 下方展开一个小表单，而不是连续弹出输入框。
- 用户可以在表单中输入：
  - gold papers：本句最应该推荐的论文。
  - acceptable papers：也可以接受的替代论文。
  - forbidden papers：不应该推荐的论文或类别描述。
  - notes：坏例子的简短说明。
- 导出的 JSONL 兼容现有 `eval/run_eval.py`。
- 文件保存到当前写作工作区：

```text
.citation-agent/eval_cases/user_bad_cases.jsonl
```

## 为什么 v7 做这个

EasyCite 的真实难点不是“能不能召回一些相关论文”，而是：

- 为什么 NeRF claim 被推荐了 3DGS-SLAM 文章？
- 为什么某篇用户知道的目标论文没有进入候选？
- 修改 prompt / rank / evidence filter 后，之前的坏例子有没有变好？

这些问题无法靠一次性手动测试解决，必须沉淀成 eval case。v7 让用户在发现坏例子的当下就能保存评测样本。

## 用户流程

```text
运行 EasyCite
↓
候选审核面板打开
↓
查看 Agent State / Run Trace / 诊断候选
↓
如果开启了 developerMode 且发现坏例子，点击 导出 Eval Bad Case
↓
在面板内展开表单，填写 gold / acceptable / forbidden / notes
↓
生成 .citation-agent/eval_cases/user_bad_cases.jsonl
```

后续可以用该文件作为 eval 输入：

```bash
.venv/bin/python eval/run_eval.py --cases /path/to/workspace/.citation-agent/eval_cases/user_bad_cases.jsonl
```

## 导出格式

每一行是一个 JSON object：

```json
{
  "id": "badcase_xxx",
  "claim": "Selected sentence...",
  "context": "Paragraph context...",
  "citation_needed": true,
  "gold_papers": ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"],
  "acceptable_papers": ["CP-SLAM: Collaborative Neural Point-based SLAM"],
  "forbidden_papers": ["pure geometric SLAM paper without neural/differentiable representation"],
  "expected_evidence_keywords": [],
  "notes": "Exported from EasyCite run ..."
}
```

## 修改文件

- `src/commands/auditSelection.ts`
  - 候选面板 `Agent State` 区域新增 `导出 Eval Bad Case`。
  - 导出 UI 从弹窗输入改为面板内折叠表单。
  - 仅在 `citation-agent.developerMode=true` 时显示和执行导出。
  - 新增 `exportEvalCaseFromSession`。

- `package.json`
  - 新增 `citation-agent.developerMode` 配置项。
  - 导出 JSONL 到当前工作区 `.citation-agent/eval_cases/user_bad_cases.jsonl`。

- `bundled/tool/citation_agent/agent_state.py`
  - schema version 升级为 `v7.agent_state.1`。

- `package.json` / `package-lock.json` / `.vscode/launch.json`
  - 版本和调试名升级为 v7。

## 与后续版本的关系

v7 的导出能力是 Bad Case Analyzer 的第一步。后续可以继续扩展：

1. 从 run record 自动预填当前推荐、诊断候选和候选生命周期。
2. UI 中直接选择某个诊断候选作为 gold paper。
3. 一键运行 eval 并展示指标变化。
4. 支持 checkpoint replay：固定 Zotero 召回结果，只重跑 metadata rank 或 LLM judge。
5. 接入 LangGraph 后，把每个 node 的 checkpoint 与 bad case 绑定。
