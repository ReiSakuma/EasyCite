# EasyCite v13 Evidence Quality 技术设计

## 目标

v13 聚焦 evidence passage 的质量控制。此前版本已经能从 Zotero、当前 BibTeX、PDF 和摘要中召回候选，并由 LLM 或本地逻辑判断支持关系；但在真实论文里，PDF 抽取出的文本经常包含参考文献列表、图表 OCR、目录页、Related Work 中转述他人工作的句子。它们语义上可能包含很多关键词，却不能证明“当前候选文献自己的观点或贡献”。

因此 v13 不再只问“这个 passage 是否相似”，而是先问“这个 passage 是否适合作为证据”。

## 核心设计

### EvidenceQuality 层

后端新增 `EvidenceQuality` 质量分析结果，挂到每个 `EvidenceSnippet` 上：

- `quality_score`: 0 到 1 的证据质量分。
- `is_own_claim`: passage 是否像作者自己的贡献/方法描述。
- `is_related_work_summary`: passage 是否像 Related Work 中总结其他论文。
- `supports_core_mechanism`: passage 是否覆盖当前 citation contract 的核心机制概念。
- `citation_density`: 文段中的引用密度。
- `quality_flags`: 可解释的质量标签。

这些字段会进入排序、LLM 判断 payload、缓存 key、候选面板 UI 和完整证据输出。

### 低质量 passage 过滤

v13 会丢弃以下 evidence：

- 参考文献列表条目。
- 图表标题、流程图文字、OCR 词流。
- TOC、front matter、投稿模板信息。
- 过短或没有句子边界的 PDF 片段。
- 高引用密度且主要在转述外部论文的 Related Work 片段。

关键原则是：不要把候选论文“引用别人”的句子误当成候选论文自己的证据。

### 排序策略

证据排序仍保留原本的 lexical/embedding 相关性分，但会乘以 `quality_score`。这意味着相似但质量差的片段会被压低，作者自有论断、摘要中的贡献句、覆盖核心机制的句子会被优先展示。

### LLM 协作方式

v13 不把质量判断完全交给 LLM，而是把 quality metadata 一并发给 LLM：

- LLM 仍负责判断 claim 与候选文献的支持关系。
- 本地 quality 层负责提醒 LLM：某个 passage 可能是引用列表、Related Work 转述、图表 OCR 或核心机制不足。
- 这样保留模型理解能力，同时减少低质量 evidence 污染。

## 用户端变化

候选面板里的 evidence 预览和展开区会显示质量提示，例如：

- `quality 0.92`
- `作者自有论断`
- `覆盖核心机制`
- `Related Work 转述`
- `核心机制不足`

用户可以直接看到系统为什么更信任某个文段，也能发现某个候选是否只是主题相似但证据质量不够。

## 验证方式

推荐使用几类句子测试：

1. 神经表示/NeRF/可微分 SLAM 相关论断。
2. 多智能体 Gaussian SLAM 全局优化论断。
3. 容易被 Related Work 误导的论断。
4. Zotero PDF 中包含流程图、参考文献列表或 arXiv front matter 的论文。

预期行为：

- 候选卡片不再优先展示参考文献列表或图表 OCR 文本。
- Related Work 中描述其他论文的句子不会作为当前候选论文的强证据。
- 作者自有贡献句会有较高 `quality_score`。
- “查看完整证据”时可以看到质量标签和完整 passage。

## 工程边界

v13 仍是启发式 quality layer，不替代 LLM 的最终语义判断。它的作用是先清洗 evidence 输入，降低脏 passage 对 LLM 和排序的干扰。后续版本可以继续把这些 quality flags 接入 Eval Harness，统计坏例中各类误证据的比例。
