# EasyCite v11: Feedback Memory

## 目标

v11 在 v10 的 eval harness 之后，补上用户侧可感知的 Memory 能力：

```text
用户接受/拒绝候选 -> 项目级反馈记忆 -> 相似 claim 中轻量加权/降权 -> trace 可解释
```

这不是把某篇论文硬编码进候选，也不会凭空召回未被检索到的论文。Memory 只对已经进入候选池的文献做小幅排序调整。

## 新增模块

- `bundled/tool/citation_agent/memory.py`
  - 使用 SQLite 保存候选反馈。
  - 默认路径：当前 workspace 的 `.citation-agent/memory.sqlite3`。
  - 支持接受 / 拒绝两种动作。
- `citationAgent/recordFeedback`
  - VS Code 前端在用户选择候选时调用。
- `citationAgent/clearMemory`
  - 清空当前项目反馈记忆。
- `citation-agent.clearProjectMemory`
  - 命令面板入口：`EasyCite: 清空项目记忆`。

## 记忆来源

候选面板中：

- 点击 `加入引用列表`：记录 accepted。
- 点击 `仅插入此条` 且插入成功：记录 accepted。
- 已选择后点击 `移出引用列表`：记录 rejected。
- `清空已选` 不记录 rejected，因为这可能只是用户重新整理选择。

## 排序策略

Memory 在 evidence verification 后、final recommendation 前生效：

```text
verified candidates
  -> evidence rank
  -> feedback memory rerank
  -> final recommendations
```

每条候选会用论文 identity 匹配历史反馈：

- DOI 优先；
- 其次规范化 title；
- 其次 citation key；
- 最后 paper id。

只有当前 claim 与历史反馈 claim 的 token overlap 达到 `memoryMinSimilarity` 时才生效。

默认参数：

```json
{
  "citation-agent.memoryEnabled": true,
  "citation-agent.memoryPositiveBoost": 0.08,
  "citation-agent.memoryNegativePenalty": 0.18,
  "citation-agent.memoryMinSimilarity": 0.22
}
```

正反馈小幅加权，负反馈略强降权；这符合用户体验：重复拒绝的弱相关文献应更快离开前排。

## 可观测性

Trace 的 `evidence_rank` 阶段会显示：

```text
memory enabled, records=3, adjustments=1
```

候选 trace 中会显示：

```text
Memory feedback: accepted, delta=+0.06, similar_claim=0.74
```

因此 memory 的影响可以被解释和复盘。

## 用户测试

1. 打开 v11 扩展开发主机。
2. 对一句 claim 运行 `EasyCite: 检查选中内容`。
3. 在候选面板中对某篇论文点击 `加入引用列表`。
4. 对相似 claim 点击灯泡里的 `重新审计此范围`。
5. 查看候选面板或 `EasyCite: 打开运行 Trace`：
   - 被接受过的同一论文如果再次进入候选池，会出现 `Memory feedback: accepted`。
   - 排名/置信度会小幅提升。
6. 执行 `EasyCite: 清空项目记忆` 后再次测试，memory 调整应消失。

## 面试表达

v11 可以这样讲：

> 我把用户审核行为接成 project-level memory。系统不会硬编码推荐某篇论文，而是在候选已经被召回并完成证据验证后，根据同项目中用户对相似 claim 的接受/拒绝反馈做轻量 rerank，并把影响写入 trace。这样既能学习用户偏好，又保持推荐链路可解释、可撤销、可回归测试。
