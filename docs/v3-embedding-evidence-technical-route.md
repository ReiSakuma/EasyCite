# EasyCite V3 Embedding Evidence Retrieval 技术路线

## 目标

V3 的目标是在 V2.5 的证据驱动引文推荐基础上，引入本地 embedding 语义检索，让 evidence passage 的召回不再过度依赖关键词字面匹配。

V2.5 已经实现：

```text
metadata 宽召回
-> abstract/PDF passage 抽取
-> 图表/参考文献/OCR 噪声过滤
-> 关键词相关度排序
-> LLM evidence judge
-> 候选面板审核
```

但仍有一个核心问题：

```text
同义表达可能无法进入 evidence shortlist
```

例如待引用文段说：

```text
Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.
```

相关论文里可能写的是：

```text
centralized pose graph refinement
global map consistency optimization
joint global bundle adjustment
offline global map fusion
```

这些表达与 claim 语义接近，但可能没有直接命中 `offline / global / optimization` 的组合。V3 要解决的就是这个 evidence 层的语义召回问题。

## 总体路线

V3 不用 embedding 替代现有规则，而是采用 Hybrid Retrieval：

```text
claim
-> metadata recall
-> PDF/abstract passage extraction
-> evidence quality filtering
-> lexical score
-> embedding score
-> hybrid rerank
-> top evidence passages
-> LLM support verification
```

核心原则：

- embedding 负责召回语义相近 passage；
- 关键词规则负责保留领域约束和可解释性；
- LLM 仍然只做最后的 evidence support 判断；
- 没有本地 embedding 模型时，系统必须自动回退到 V2.5 的关键词检索。

## 模型选择

V3 首选轻量本地 embedding 模型，CPU 可运行，不依赖用户提供 API。

候选模型：

```text
BAAI/bge-small-en-v1.5
BAAI/bge-base-en-v1.5
thenlper/gte-small
intfloat/e5-small-v2
```

第一版建议使用：

```text
BAAI/bge-small-en-v1.5
```

原因：

- 科研 PDF evidence 主要是英文；
- 模型体积相对较小；
- CPU 推理速度可接受；
- 语义检索效果比纯关键词更稳；
- ModelScope 上通常可获取。

后续可以增加多语种模型选项，但 V3 第一版不需要把模型选择做得过度复杂。

## 运行时方案

优先考虑 ONNX Runtime，而不是直接依赖 PyTorch。

推荐结构：

```text
sentence-transformers 原模型
-> 预转换 ONNX 或下载 ONNX 版本
-> onnxruntime CPU 推理
```

原因：

- VS Code 扩展分发不适合强依赖 PyTorch；
- PyTorch 包体积大，安装慢，用户环境差异多；
- ONNX Runtime CPU 更适合桌面插件场景；
- 模型文件可首次使用时下载，不打包进扩展。

V3 初期可以先实现一个可插拔接口：

```python
class Embedder:
    def available(self) -> bool: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
```

底层先允许两种实现：

```text
LocalOnnxEmbedder
DisabledEmbedder
```

后续如需快速原型，也可以临时接入 sentence-transformers，但成品扩展应以 ONNX 为主。

## 模型下载

模型不随扩展一起打包。首次使用 embedding evidence retrieval 时：

```text
检查本地模型缓存
-> 不存在则询问用户是否下载
-> 从 ModelScope 下载
-> 校验必要文件
-> 写入模型 manifest
-> 后续直接复用
```

缓存目录建议：

```text
~/.cache/citation-agent/models/bge-small-en-v1.5/
```

或者在 VS Code 环境中使用：

```text
context.globalStorageUri/models/
```

考虑到 Python 后端更容易访问文件系统，第一版建议使用：

```text
~/.cache/citation-agent/models/
```

需要保存：

```json
{
  "model_id": "BAAI/bge-small-en-v1.5",
  "revision": "...",
  "backend": "onnxruntime",
  "downloaded_at": "...",
  "files": {
    "model": "model.onnx",
    "tokenizer": "tokenizer.json"
  }
}
```

用户体验要求：

- 不静默下载模型；
- 下载前提示模型大小和来源；
- 下载失败时回退关键词检索；
- 支持命令 `EasyCite: 下载/检查本地 Embedding 模型`；
- 支持命令 `EasyCite: 清理本地 Embedding 模型缓存`。

## 新增配置

建议新增 VS Code 配置：

```json
{
  "citation-agent.embeddingEnabled": true,
  "citation-agent.embeddingModelProvider": "fastembed",
  "citation-agent.embeddingModelId": "BAAI/bge-small-en-v1.5",
  "citation-agent.embeddingCacheDir": "",
  "citation-agent.embeddingAutoDownload": "ask",
  "citation-agent.embeddingTopK": 12,
  "citation-agent.hybridLexicalWeight": 0.35,
  "citation-agent.hybridEmbeddingWeight": 0.65
}
```

含义：

```text
embeddingEnabled        是否启用 embedding 检索
embeddingModelProvider  模型后端，当前可运行实现为 fastembed
embeddingModelId        模型 ID
embeddingCacheDir       空值表示使用默认缓存目录
embeddingAutoDownload   ask / never / always
embeddingTopK           embedding 召回 passage 数量
hybridLexicalWeight     关键词分数权重
hybridEmbeddingWeight   向量相似度权重
```

默认应为：

```text
embeddingEnabled = false 或 ask-on-first-use
```

原因是首次模型下载涉及网络和磁盘空间，不应该在用户不知情时发生。

## 后端模块设计

新增目录：

```text
bundled/tool/citation_agent/embedding/
├── __init__.py
├── model_manager.py
├── embedder.py
├── passage_index.py
└── hybrid_ranker.py
```

### model_manager.py

职责：

```text
解析 embedding 配置
检查本地模型是否存在
触发 ModelScope 下载
校验模型文件
返回模型路径
```

核心接口：

```python
class EmbeddingModelManager:
    def model_available(self) -> bool: ...
    def ensure_model(self, allow_download: bool) -> ModelStatus: ...
    def model_path(self) -> Path | None: ...
```

`ModelStatus` 应包含：

```json
{
  "available": true,
  "downloaded": false,
  "model_id": "...",
  "path": "...",
  "reason": "..."
}
```

### embedder.py

职责：

```text
加载 tokenizer 和 ONNX 模型
对 claim / passage 生成 embedding
做 L2 normalize
```

核心接口：

```python
class LocalEmbedder:
    def embed_query(self, text: str) -> list[float]: ...
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
```

注意：

- query 可以加模型推荐前缀，例如 BGE:

```text
Represent this sentence for searching relevant passages: ...
```

- passage 直接使用原文；
- 输出向量统一归一化，余弦相似度可简化为点积。

### passage_index.py

职责：

```text
缓存 PDF/abstract passage embedding
避免同一 PDF 每次重复计算
```

缓存 key：

```text
paper_id
pdf_path
pdf_mtime
page
passage_hash
model_id
model_revision
```

建议用 SQLite：

```text
~/.cache/citation-agent/embeddings/index.sqlite3
```

表结构：

```sql
CREATE TABLE passage_embeddings (
  cache_key TEXT PRIMARY KEY,
  paper_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  page INTEGER,
  section TEXT,
  embedding BLOB NOT NULL,
  created_at TEXT NOT NULL
);
```

第一版也可以先用 JSONL/npz，但 SQLite 更适合长期缓存和增量更新。

### hybrid_ranker.py

职责：

```text
融合 lexical score 和 embedding score
输出最终 evidence passage 排序
```

建议公式：

```text
hybrid_score =
  lexical_weight * normalized_lexical_score
  + embedding_weight * normalized_embedding_score
  + source_bonus
  - quality_penalty
```

默认：

```text
lexical_weight = 0.35
embedding_weight = 0.65
```

source bonus：

```text
abstract +0.03
pdf正文 +0.00
metadata -0.10
```

quality penalty：

```text
reference/caption/OCR 直接过滤，不进入排序
metadata-only 低置信降权
```

## EvidenceRetriever 改造

V2.5 当前逻辑：

```text
collect passages
-> quality filter
-> lexical score
-> sort by lexical score
-> top maxEvidencePassages
```

V3 改成：

```text
collect passages
-> quality filter
-> lexical score
-> embedding score if available
-> hybrid score
-> deterministic rerank
-> top maxEvidencePassages
```

如果 embedding 不可用：

```text
fallback to V2.5 lexical ranking
```

`EvidenceSnippet.score` 可以暂时继续保存 hybrid score。后续可扩展 schema：

```json
{
  "score": 1.23,
  "lexical_score": 0.71,
  "embedding_score": 0.84,
  "ranker": "hybrid_embedding"
}
```

为了兼容前端，第一版可以只在 reason/log 中输出细节，不立即改协议 schema。

## LLM Judge 的职责边界

embedding 只负责“找哪些 passage 值得读”，不负责最终决定是否引用。

最终判断仍由：

```text
LLM evidence judge
或本地 evidence rule fallback
```

完成。

LLM 输入仍然限制为：

```text
最多 5 passage / candidate
最多 8 candidate
最多 40 passage
```

也就是说 embedding 不会增加 LLM token 成本，反而会提高送给 LLM 的 passage 质量。

## 前端交互设计

V3 不需要大改候选面板，但应增加可解释性。

候选 evidence 显示建议：

```text
最相关文段 (hybrid · score 0.82 · embedding 0.78 · lexical 0.51)
```

如果 embedding 未启用：

```text
最相关文段 (lexical · score 1.45)
```

首次需要模型时：

```text
EasyCite 需要下载本地 embedding 模型以启用语义 evidence 检索。
模型来源：ModelScope
模型：BAAI/bge-small-en-v1.5
预计大小：...

[下载并启用] [本次跳过] [始终使用关键词检索]
```

命令面板新增：

```text
EasyCite: 检查本地 Embedding 模型
EasyCite: 下载本地 Embedding 模型
EasyCite: 清理 Embedding 缓存
```

## 失败与降级

任何 embedding 环节失败，都不能阻断引用推荐。

降级策略：

```text
模型不存在 -> 询问下载；用户拒绝则 lexical fallback
下载失败 -> lexical fallback + 输出日志
模型加载失败 -> lexical fallback + 输出日志
embedding 计算失败 -> 当前候选 lexical fallback
cache 写入失败 -> 不影响本次检索
```

日志应明确显示：

```text
Embedding retrieval enabled=false reason=model_not_found
Embedding model download skipped by user
Embedding model loaded: ...
Embedding-ranked passages: ...
Hybrid-ranked passages: ...
```

## 隐私与分发

V3 embedding 是本地模型：

- PDF 文本不会发送给 embedding API；
- 只有 LLM evidence judge/translation 使用用户配置的外部模型 API；
- ModelScope 下载只发生在用户明确允许时；
- 模型缓存可删除；
- 扩展包不携带大模型文件。

这符合成品扩展逻辑：

```text
扩展轻量分发
首次启用高级语义检索时下载模型
无模型时仍可使用基础功能
```

## 测试计划

单元测试：

```text
model_manager:
  - 模型存在/不存在状态判断
  - autoDownload=never 时不触发下载
  - 下载失败返回 fallback 状态

embedder:
  - embedding 维度正确
  - normalize 后向量范数约等于 1

passage_index:
  - 相同 passage 命中缓存
  - PDF mtime 变化后缓存失效

hybrid_ranker:
  - 同义表达 passage 被排到关键词弱匹配 passage 前
  - 图表/参考文献噪声不会进入 ranking
  - embedding 不可用时回退 lexical 排序
```

集成测试：

```text
claim: Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.

候选 passage A:
  contains exact words offline/global/optimization

候选 passage B:
  says centralized pose graph refinement for globally consistent multi-agent mapping

预期:
  B 即使关键词不完全匹配，也能通过 embedding 进入 evidence top-k
```

## 当前实现状态

已完成 V3 第一阶段真实 embedding 接入：

```text
配置层:
  - 后端新增 EmbeddingConfig
  - 前端 settings/env 新增 embedding 与 hybrid weight 配置
  - 默认 provider 改为 fastembed

检索层:
  - 新增 bundled/tool/citation_agent/embedding/
  - 新增 hybrid_ranker.py
  - 新增 FastEmbedder，使用 fastembed + ONNX Runtime CPU 推理
  - EvidenceRetriever 已改为调用 hybrid evidence ranker
  - 新增 PDF passage SQLite 缓存，按 pdf path + mtime + size 失效
  - 新增 embedding vector SQLite 缓存，按 model_id + text_hash 复用
  - LLM evidence judge 已改为批量判断 shortlist 候选，减少多次串行 API 往返
  - 默认 embeddingEnabled=false，行为回退到 V2.5 关键词排序
  - embeddingEnabled=true 且 provider=fastembed 时使用真实 BAAI/bge-small-en-v1.5 向量
  - metadata 在存在 abstract/PDF passage 时降为兜底，避免标题/作者词面命中压过正文证据

测试层:
  - 新增 deterministic fake embedder 测试
  - 验证 hybrid ranker 可以把语义更接近的 passage 排到词面命中更高的 passage 前
  - 验证 embedding disabled 时保持原 lexical fallback 顺序
  - 验证 LLM evidence judge 候选数量限制和批量调度
  - 本机已下载并 smoke test BAAI/bge-small-en-v1.5，向量维度 384
```

尚未完成的 V3 后续工作：

```text
VS Code 下载/检查/清理 embedding 模型命令
候选面板中展示 lexical_score / embedding_score / hybrid_score 明细
ModelScope 专用下载器
```

因此当前 V3 已经可以开始做真实 embedding evidence 检索测试。需要注意的是，当前模型下载由 fastembed/HuggingFace 生态完成；ModelScope 专用下载器仍是下一阶段。

手动测试：

```text
1. 确认 v3/.venv 已安装 requirements-embedding.txt
2. 确认 BAAI/bge-small-en-v1.5 已缓存到 ~/.cache/citation-agent/embeddings
3. 在 VS Code 设置中启用 embeddingEnabled=true
4. 设置 embeddingModelProvider=fastembed
5. 设置 embeddingAutoDownload=never 或 ask，避免后端静默下载
6. 运行 EasyCite: 检查选中内容
7. 查看 EasyCite 输出日志中的 Embedding ranker status / Embedding ranker applied
8. 对比 evidence passage 是否优先来自 abstract/PDF 正文，而不是 metadata
```

## 实施步骤

建议按 5 个小步实现：

### Step 1: 配置和文档

- 增加 embedding 配置项；
- 增加模型状态命令；
- 不实现下载，只返回 unavailable；
- 后端仍走 V2.5 lexical fallback。

### Step 2: Passage embedding 接口

- 新增 embedding module；
- 实现 `DisabledEmbedder` 和接口；
- 写 hybrid ranker 单元测试；
- 暂时用 fake embedding 测试排序。

### Step 3: fastembed ONNX 模型接入

- 安装 `fastembed` 可选依赖；
- 使用 `BAAI/bge-small-en-v1.5`；
- 缓存到 `~/.cache/citation-agent/embeddings`；
- 支持 `autoDownload=never/ask` 时只读本地缓存；
- 模型缺失或加载失败时回退 lexical。

### Step 4: ONNX CPU 推理

- 通过 fastembed 接入 tokenizer + ONNX Runtime；
- 实现 query/passages embedding；
- 加入 passage embedding SQLite 缓存。

### Step 5: EvidenceRetriever 集成

- 将 lexical ranker 替换为 hybrid ranker；
- 输出 ranker 类型和 score 细节；
- 更新候选面板 evidence preview；
- 完整回归测试。

## V3 第一版完成标准

V3 第一版不要求 embedding 功能完美，但必须满足：

```text
1. v3 从 v2 独立出来，不影响 v2 可用性；
2. 技术路线清晰，配置项明确；
3. embedding 不可用时稳定回退 V2.5；
4. 有 fake embedding 的单元测试证明 hybrid ranker 可召回同义 passage；
5. 模型下载必须经过用户确认；
6. 不把模型文件打包进扩展。
```

完成这些后，再进入 passage embedding 缓存、VS Code 模型管理命令和 ModelScope 专用下载器实现。
