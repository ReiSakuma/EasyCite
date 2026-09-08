# EasyCite

EasyCite 是一款面向 LaTeX 科研写作的 VS Code 扩展。它可以识别可能缺少引用的论断，将当前 BibTeX 工程和指定 Zotero Collection 组织为本地 RAG 语料库，通过论文级与段落级混合检索找到候选及证据，并由外部 LLM 判断候选文献是否真正支持原句。

EasyCite 不会直接替用户决定引用。所有候选、证据和 BibTeX 写入计划都会先进入审核界面，由用户确认后再修改 TeX 与 `.bib` 文件。

## 主要功能

- 自动扫描 TeX 工程，在疑似缺少引用的句子下显示黄色波浪线和灯泡操作。
- 检查选中的 Claim，并结合段落上下文理解待引用观点。
- 将当前 `.bib` 与所选 Zotero Collection 同步为项目私有、可增量更新的本地 RAG 语料库。
- 在论文级和段落级分别执行 BM25 + Dense 双路召回，并通过 RRF 融合与 Cross-Encoder 精排。
- 先按 PDF 版面、列位置、行距和缩进重建自然段，再以自适应语义边界生成 parent-child chunk，保留证据页码与上下文。
- 使用 LLM 完成查询规划、证据验证和中英对照，区分直接支持、部分支持、基础性支持与证据不足。
- 在候选审核界面比较论文、证据、页码、DOI、来源和 Citation Key。
- 支持多选引用，在英文句末标点前插入 `\cite{...}`。
- 通过 Better BibTeX 导出条目，写入工程已有 `.bib`；没有 `.bib` 时才创建新文件。
- 检查重复 Citation Key、错误 BibTeX、来源可信度，并在写入后进行校验。
- 基于 LangGraph StateGraph 执行工作流，支持 checkpoint、取消、恢复、缓存和结构化 Trace。
- 提供设置向导、工具健康检查、失败降级、重试、熔断、Eval Harness 与项目级反馈记忆。

## 工作流程

```text
TeX Claim
  -> Claim 理解
  -> LLM 查询规划
  -> 当前 BibTeX / Zotero Collection 语料同步
  -> 标题与 DOI 去重
  -> 论文级 BM25 + Dense + RRF 召回
  -> 为高相关论文懒加载并增量索引 PDF
  -> 段落级 BM25 + Dense + RRF 召回
  -> Cross-Encoder Evidence 精排
  -> LLM 证据验证
  -> 用户审核候选
  -> 插入 \cite{} 与 BibTeX
  -> 写后校验
```

## 环境要求

| 组件 | 要求 |
| --- | --- |
| VS Code | 1.78 或更高版本 |
| Python | Windows 推荐 3.11 至 3.13；当前通用/Linux VSIX 要求 Python 3.13 |
| LaTeX 工程 | 至少包含一个 `.tex` 文件；推荐使用 BibLaTeX 或标准 BibTeX |
| Zotero | 可选；使用 Zotero 候选时需要桌面版 Zotero 和本地 API |
| Better BibTeX | 可选但推荐；用于稳定 Citation Key 和 BibTeX 导出 |
| 模型 API | 可选但推荐；支持 OpenAI、DeepSeek、OpenRouter 和 OpenAI-Compatible 服务 |
| 网络 | 首次安装 Python 依赖、调用外部 LLM、首次下载 Embedding 模型时需要 |

没有 Zotero 时，EasyCite 仍可为当前 `.bib` 建立本地 RAG；Embedding 不可用时降级为 BM25，PDF 不可用时使用摘要和元数据，模型 API 不可用时使用本地证据规则。降级后仍可运行，但语义召回和证据判断质量会下降。

## 安装扩展

### 1. 选择安装包

在 GitHub Releases 下载与环境匹配的文件：

- Windows 本地 VS Code：`easycite-v22-Windows.vsix`
- Linux 或运行在 WSL Remote Extension Host 中的 VS Code：`easycite-v22.vsix`

如果以前安装过旧版扩展，请先卸载旧扩展，避免两个版本注册相同的兼容命令。

### 2. 从 VSIX 安装

在 VS Code 中：

1. 打开左侧“扩展”。
2. 点击扩展面板右上角的 `...`。
3. 选择“从 VSIX 安装...”。
4. 选择下载的 `easycite-v22*.vsix`。
5. 安装完成后点击“重新加载”。

也可以在终端安装：

```bash
code --install-extension easycite-v22.vsix
```

Windows PowerShell：

```powershell
code --install-extension .\easycite-v22-Windows.vsix
```

### 3. WSL 用户注意事项

如果 LaTeX 工程在 WSL 中，请在 VS Code 左下角显示 `WSL: ...` 的窗口中安装通用/Linux VSIX。Python 也必须安装在 WSL 中。Zotero 可以运行在 Windows，EasyCite 会通过 `127.0.0.1:23119` 访问其本地 API。

## 首次设置向导

打开任意包含 `.tex` 文件的工作区，然后按 `Ctrl+Shift+P`，运行：

```text
EasyCite: 设置向导
```

推荐选择“完整设置”。向导按以下顺序执行。

### 第一步：Python 与依赖安装策略

1. 选择“缺依赖时询问”或“缺依赖时自动安装”。
2. EasyCite 自动检测当前 VS Code 环境中的 Python。
3. 缺少 `pydantic`、`pygls`、`lsprotocol` 或 `langgraph` 时，允许向导安装。
4. 依赖安装到 VS Code 的 EasyCite 私有目录，不会写入 Conda Base 或系统 Python。

如果未检测到 Python：

- Windows 安装 Python 或 Miniconda，并确保 `python.exe` 可用。
- Linux/WSL 安装 Python 3.13，并重新加载 VS Code 窗口。
- 高级用户可设置 `citation-agent.interpreter` 指定解释器。

### 第二步：Zotero 与 Better BibTeX

如果需要检索个人文献库，选择“启用 Zotero + Better BibTeX”。默认地址保持：

```text
http://127.0.0.1:23119/api
```

#### Zotero 设置

1. 安装并启动 Zotero 桌面版。
2. 打开 Zotero“设置 -> 高级”。
3. 勾选“允许此计算机上的其他应用程序与 Zotero 通讯”。
4. 保持 Zotero 在后台运行。

可使用下面的命令验证 Local API。

Windows PowerShell：

```powershell
curl.exe --noproxy "*" -i "http://127.0.0.1:23119/api/users/0/items/top?limit=1&format=json" -H "Zotero-API-Version: 3"
```

Linux/WSL：

```bash
curl --noproxy '*' -i \
  'http://127.0.0.1:23119/api/users/0/items/top?limit=1&format=json' \
  -H 'Zotero-API-Version: 3'
```

正确状态应为 `HTTP 200`，并返回至少一个 JSON 条目。

#### Better BibTeX 设置

1. 从 Better BibTeX 官方 GitHub Releases 下载 `.xpi`。
2. 在 Zotero 中打开“工具 -> 插件”。
3. 点击齿轮按钮，选择“Install Add-on From File...”。
4. 选择 `.xpi` 并重启 Zotero。
5. 重新运行 `EasyCite: 设置向导`，确认 Zotero 和 Better BibTeX 均可访问。

Better BibTeX 会为 Zotero 条目维护稳定的 Citation Key。EasyCite 查询到候选后，会按需调用 Better BibTeX 导出条目，不需要手动为整个文库逐篇导出。

### 第三步：模型 API

选择模型 Provider：

- OpenAI
- DeepSeek
- OpenRouter
- OpenAI-Compatible

随后依次填写模型名称、可选 API Base URL 和 API Key。API Key 使用 VS Code SecretStorage 保存，不会写入 `settings.json`、Trace 或项目文件。

模型命名示例：

```text
gpt-4o-mini
deepseek/deepseek-chat
openrouter/openai/gpt-4o-mini
```

配置完成后，EasyCite 会重启 Python 后端。也可以稍后单独运行：

```text
EasyCite: 配置模型服务
```

### 第四步：本地 RAG 与 Embedding 模型

推荐选择“启用本地 RAG 语义检索”，保持默认模型：

```text
BAAI/bge-small-en-v1.5
```

EasyCite 使用 FastEmbed 与 ONNX 在本地 CPU 上运行。首次使用会下载模型到配置的缓存目录，耗时取决于网络；后续运行直接使用缓存。

首次检索会读取所选 Collection 的论文元数据和摘要，建立 `.citation-agent/rag.sqlite3`。论文级召回完成后，默认只为最相关的 12 篇论文定位并索引 PDF，避免用户等待全库 PDF 解析。索引记录论文/PDF 指纹，后续查询会复用已有内容，并按需逐步补充新的高相关 PDF。点击“重新审计”会重新执行查询规划、双路召回、RRF、精排和 LLM 验证，但不会无意义地重建未变化的 PDF 索引。

RAG 的默认流程为：论文级 BM25 + Dense 召回、RRF 融合、段落级 BM25 + Dense 召回、RRF 融合、Cross-Encoder 精排、LLM 证据验证。Embedding 或精排模型不可用时，Trace 会明确显示原因，并降级到 BM25 或确定性本地 reranker。

高级设置 `citation-agent.ragPdfIndexTopK` 控制单次前台查询最多新增索引多少篇 PDF，默认值为 `12`。减小它可以降低首次等待时间，但更多候选会暂时只使用摘要证据。

`hash embedding` 只用于离线开发测试，不具备真实语义检索能力。

## RAG 检索逻辑

EasyCite 的 RAG 不会把整篇论文直接发送给 LLM，也不会只根据标题做一次简单排序。一次候选检查分为以下阶段：

1. **语料范围同步**：读取当前工程 `.bib` 和用户选中的 Zotero Collection。Collection 决定允许参与检索的文献范围。
2. **增量索引**：将标题、作者、摘要及已有 PDF Chunk 写入项目私有 SQLite/FTS5 索引；通过元数据、PDF 修改时间、文件大小和模型版本判断是否需要更新。
3. **论文级召回**：针对 Claim 与 LLM 查询规划结果，分别执行 BM25 词法召回和本地 Dense Embedding 召回，再用 RRF 融合两个排名。
4. **PDF 懒加载**：论文级召回后，默认只为前 12 篇高相关论文定位 Zotero PDF 并建立内容索引，避免每次等待全库 PDF 解析。已经索引的 PDF 会在后续查询中复用。
5. **段落级召回与精排**：在候选论文的摘要和 PDF Chunk 中再次执行 BM25 + Dense + RRF，并使用 Cross-Encoder 对少量高相关段落精排。系统先依据列位置、行距和缩进将 PDF 文本块重建为自然段，再结合固定阈值与章节内相似度低谷识别主题边界；短排版碎片与邻近正文共享语义编码，但原文仍完整保留。Child overlap 不会跨越语义 Parent。
6. **证据验证**：只把最终筛出的少量 Evidence Passage、Claim 和必要上下文交给 LLM，区分直接支持、基础性支持、部分支持、仅相关和证据不足。

RAG 索引默认保存在当前工程：

```text
.citation-agent/rag.sqlite3
```

“查看候选引用”只打开上一次审核结果；“重新审计此范围”会重新执行查询规划、混合召回、精排和 LLM 验证，但会复用未变化的本地索引。

### 第五步：PDF 读取

选择“安装 PDF 阅读增强依赖”。向导会在扩展私有目录安装 PyMuPDF。启用后，EasyCite 可以读取 Zotero PDF 附件并显示证据页码；跳过时仍可使用标题、摘要和 BibTeX 元数据。

### 第六步：检查工具状态

向导结束后运行：

```text
EasyCite: 打开工具健康检查
```

确认 Python、Zotero、Better BibTeX、LLM、Embedding、PDF 和 BibTeX Writer 的状态。部分可选组件不可用时，页面会显示降级方式和修复建议。

## 使用方法

### 自动扫描 TeX 工程

打开主 `.tex` 文件后运行：

```text
EasyCite: 扫描当前 TeX 工程引用标记
```

EasyCite 会解析主 TeX、`\input`/`\include` 文件和 bibliography 声明，在疑似缺少引用的句子下显示黄色波浪线。点击灯泡可以：

- 查看已有候选引用
- 重新审计此范围
- 审计此句并推荐引用
- 清除此标记

“查看候选引用”打开上一次审核结果；“重新审计”会跳过结果缓存并重新运行检索与验证。

### 检查选中内容

1. 在 `.tex` 文件中选中一个需要引用的完整句子或短段落。
2. 按 `Ctrl+Shift+P`。
3. 运行 `EasyCite: 检查选中内容`。
4. 等待查询规划、RAG 论文召回、PDF Passage 检索与 LLM 验证完成。

### 审核候选文献

候选页面会展示：

- 论文标题、作者、年份、DOI、来源与 Citation Key
- 直接支持、部分支持、基础性支持或证据不足标签
- LLM 判断理由和置信度
- 最相关 Evidence Passage、PDF 页码与中英对照
- 未进入主推荐区的诊断候选及未入选原因

点击“查看完整证据”会在当前页面展开内容。可以逐篇查看后加入引用列表，也可以直接只插入某一篇。

### 插入多篇引用

1. 对需要的候选点击“加入引用列表”。
2. 检查页面顶部的已选引用。
3. 点击“插入已选择的引用”。
4. 确认 TeX 路径、目标 `.bib` 和待写入条目。

EasyCite 会将英文引用插入句末标点之前，例如：

```latex
Existing methods rely on offline global optimization \cite{paperA,paperB}.
```

如果工程已经声明 `references.bib`，条目会写入该文件；只有没有找到可用 `.bib` 时才创建新文件。

### 限定 Zotero 文献范围

运行：

```text
EasyCite: 配置 Zotero 文献范围
```

选择一个或多个 Collection 后，后续 RAG 语料只包含这些 Collection 和当前工程 `.bib` 中的论文。切换到“全部文库”可以恢复全库同步。Collection 或 PDF 发生变化时，下一次检索会自动增量更新索引。

### 项目反馈记忆

EasyCite 会在项目范围记录用户接受或拒绝候选的反馈，用于后续排序。运行下面的命令可以清空当前项目记忆：

```text
EasyCite: 清空项目记忆
```

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `EasyCite: 设置向导` | 检查并配置全部依赖 |
| `EasyCite: 打开工具健康检查` | 查看各工具状态和降级路径 |
| `EasyCite: 配置模型服务` | 配置 Provider、模型和 API Key |
| `EasyCite: 配置 Zotero 文献范围` | 限定检索 Collection |
| `EasyCite: 检查选中内容` | 审计当前选中 Claim |
| `EasyCite: 扫描当前 TeX 工程引用标记` | 自动扫描工程并生成黄色诊断 |
| `EasyCite: 返回候选审核` | 回到最近一次候选页面 |
| `EasyCite: 清除全部引用标记` | 清除当前工程诊断 |
| `EasyCite: 清空项目记忆` | 删除当前项目反馈记忆 |

开发者命令默认隐藏。将 `citation-agent.developerMode` 设置为 `true` 后，可以使用运行 Trace、Agent State、Run Record 和 Eval Bad Case 导出。

## 高级工作区设置

大多数用户只需使用设置向导。需要手动配置时，可在工作区 `.vscode/settings.json` 中填写：

```json
{
  "citation-agent.zoteroEnabled": true,
  "citation-agent.zoteroEndpoint": "http://127.0.0.1:23119/api",
  "citation-agent.ragEnabled": true,
  "citation-agent.ragMaxDocuments": 500,
  "citation-agent.ragChildChunkTokens": 280,
  "citation-agent.ragChunkOverlapTokens": 48,
  "citation-agent.ragParentChunkTokens": 900,
  "citation-agent.ragSemanticChunkingEnabled": true,
  "citation-agent.ragSemanticBoundaryThreshold": 0.45,
  "citation-agent.ragSemanticHardBoundaryThreshold": 0.30,
  "citation-agent.ragSemanticMinParentTokens": 180,
  "citation-agent.ragSemanticAdaptiveEnabled": true,
  "citation-agent.ragSemanticAdaptivePercentile": 0.20,
  "citation-agent.ragSemanticAdaptiveMaxThreshold": 0.72,
  "citation-agent.ragSemanticMinScoreDrop": 0.08,
  "citation-agent.ragSemanticContextWindow": 2,
  "citation-agent.ragSemanticEmbeddingMinTokens": 24,
  "citation-agent.ragPaperFusedTopK": 30,
  "citation-agent.ragPdfIndexTopK": 12,
  "citation-agent.ragPassageFusedTopK": 50,
  "citation-agent.ragRerankerEnabled": true,
  "citation-agent.ragRerankerModelId": "Xenova/ms-marco-MiniLM-L-6-v2",
  "citation-agent.maxSuggestions": 5,
  "citation-agent.maxAbstractCandidates": 8,
  "citation-agent.maxEvidencePassages": 5,
  "citation-agent.enableEvidenceVerification": true,
  "citation-agent.embeddingEnabled": true,
  "citation-agent.embeddingModelProvider": "fastembed",
  "citation-agent.embeddingModelId": "BAAI/bge-small-en-v1.5",
  "citation-agent.embeddingAutoDownload": "ask"
}
```

`citation-agent.*` 是为兼容旧版本保留的内部配置命名空间。产品界面与命令名称均为 EasyCite。不要把 API Key 写入 JSON，请通过 `EasyCite: 配置模型服务` 保存。

## 常见问题

### Zotero 返回 502

通常是本机代理拦截了 `127.0.0.1`。将 `127.0.0.1`、`localhost` 加入 `NO_PROXY`，或使用前面的 `--noproxy '*'` 命令验证。

### Zotero 返回 500

先完全退出 Zotero，再以“帮助 -> 故障排除模式”重启并测试 Local API。如果恢复为 200，再正常重启。问题通常来自 Zotero 本地 Connector API 的临时初始化状态。

### Zotero 正常但 Better BibTeX 不可用

确认 Better BibTeX 已安装且启用，重启 Zotero，再运行设置向导。Local API 和 Better BibTeX JSON-RPC 是两个独立检查项。

### 向导认为 Python 包没有安装

运行 `EasyCite: 打开工具健康检查`，确认 EasyCite 使用的解释器。依赖必须安装在扩展检测到的 Python 或扩展私有目录中，而不是另一个 Conda 环境。

### 第一次 RAG 检索很慢

首次运行可能需要下载并初始化 Embedding 与 Cross-Encoder ONNX 模型，同时为首批高相关 PDF 建立 Chunk 和向量。v0.22 会先重建真实 PDF 自然段，再用整篇批量 Embedding 和章节内自适应低谷识别主题边界；短碎片会合并参与语义编码，Child overlap 只发生在同一语义 Parent 内。后续会复用模型缓存与 `.citation-agent/rag.sqlite3`。开发者模式下可打开运行 Trace，分别检查 `zotero_retrieval`、`rag_index_sync`、`rag_paper_hybrid_retrieval`、`rag_pdf_index`、`rag_semantic_chunking`、`rag_passage_hybrid_retrieval`、`rag_reranker` 和 `rag_total` 的耗时。

如果 `rag_pdf_index` 持续占用较长时间，可降低 `citation-agent.ragPdfIndexTopK`；如果 `evidence_verification` 最慢，则瓶颈来自外部 LLM，而不是本地 RAG。

### LLM 没有生效

重新运行 `EasyCite: 配置模型服务`，确认 API Key、模型名和 API Base。开发者模式下可以打开运行 Trace，检查 `query_planning` 是否显示 `source=llm`，以及 Evidence Verification 的 LLM 请求数量。

## 隐私与数据

- API Key 保存在 VS Code SecretStorage，不写入仓库或运行记录。
- Zotero 元数据与 PDF 默认在本机读取。
- 启用外部模型后，Claim、必要上下文和筛选后的短 Evidence Passage 会发送到所配置的模型服务。
- EasyCite 使用上下文预算压缩输入，不会无条件上传整篇论文。
- 引用和 BibTeX 写入前必须由用户确认。

## 从源码开发

```bash
git clone https://github.com/ReiSakuma/EasyCite.git
cd EasyCite

npm ci
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.in -r requirements-dev.txt

npm run compile
python -m pytest src/test/python_tests
```

在 VS Code 中打开仓库，选择 `Debug EasyCite` 调试配置并按 `F5`，会启动扩展开发宿主。

### 构建 VSIX

通用/Linux：

```bash
npm run vendor-python-runtime
npm run vsce-package
```

Windows：

```bash
npm run vsce-package-windows
```

输出文件：

- `easycite-v22.vsix`
- `easycite-v22-Windows.vsix`

## 项目结构

```text
src/                         VS Code 扩展与 Webview
bundled/tool/citation_agent/ Python Agent、LangGraph、RAG 与证据验证
protocol/                    LSP/JSON Schema 协议
eval/                        Eval Harness 与测试用例
docs/                        各版本技术设计
scripts/                     依赖打包与 Mock Zotero 工具
src/test/python_tests/       Python 回归测试
```

## License

本项目采用 [MIT License](LICENSE)。
