# EasyCite v2.5 Technical Design

## 目标

V2.5 在 V2“本地候选引用推荐”之上，按
`证据驱动引文推荐技术路线_V3起.md` 引入证据验证链路：

```text
LaTeX 选中文本
-> 判断是否需要引用
-> 构造 claim-aware 检索短语
-> 宽召回当前 .bib + Zotero 候选
-> 对前 8 篇候选抽取摘要/PDF evidence passages
-> 判断 passage 是否真正支持选中文段
-> 返回带 verdict 的候选论文
-> 用户审核后插入 \cite{key}
```

这版仍保留 V2 的轻量架构，但已经不再把“元数据相关”直接当作“可引用证据”。默认最多召回 50 个元数据候选，只让前 8 篇进入证据验证，每篇最多读取 5 个短 passage。

## 新增能力

- 自动发现当前 TeX 文件引用的 BibTeX 文件。
- 当 TeX 文件没有显式声明 `.bib` 时，扫描工作区内的 `.bib`。
- 解析 BibTeX entry 的 title、author、year、venue、abstract、doi、url。
- 根据选中文本关键词与 BibTeX 元数据重叠度排序候选。
- 增加可配置 Zotero 只读 HTTP adapter。
- Zotero 官方 Local API 不再只用整句查询，会生成多条短查询，例如 `multi-agent Gaussian SLAM`、`Gaussian SLAM global optimization`、`Gaussian SLAM`。
- 过滤 Zotero attachment、PDF 附件等非论文条目。
- 支持从 Zotero `links.attachment` / `links.enclosure` 解析本地 PDF 路径；Windows file URI 会转换成 WSL `/mnt/c/...` 路径。
- 从 `.bib` abstract、Zotero abstract 和可选本地 PDF 中抽取 evidence passages。
- 对候选输出 `direct_support`、`partial_support`、`related_only`、`insufficient_evidence` 等 verdict。
- 配置了模型 API 时，可让 LLM 阅读 passage 并做证据支持性判断；未配置时回退本地规则。
- `EasyCite: 检查选中内容` 返回候选数量，并可打开 QuickPick 审核候选。
- 用户确认后，可以把有 citation key 的候选插入为 `~\cite{key}`。

## 后端设计

新增模块：

```text
bundled/tool/citation_agent/
├── bibliography.py
├── claim_analysis.py
├── evidence.py
├── recommender.py
└── zotero.py
```

### BibRepository

`BibRepository` 负责：

1. 从当前 `.tex` 文件解析：
   - `\addbibresource{...}`
   - `\bibliography{...}`
2. 如果没有找到显式引用，则递归扫描工作区 `.bib`。
3. 跳过 `.git`、`.venv`、`node_modules`、`dist` 等工程目录。
4. 解析 BibTeX entry。
5. 用关键词重叠计算候选置信度。

### ClaimProfile

`claim_analysis.py` 把选中文段转换为可调试的检索计划：

```text
claim
-> terms
-> field phrases
-> support terms
-> query phrases
```

例如：

```text
Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.
```

会生成类似：

```text
multi-agent Gaussian SLAM
Gaussian SLAM global optimization
collaborative Gaussian SLAM
multi-agent SLAM global consistency
Gaussian SLAM
```

这样避免把整句直接丢给 Zotero 导致 0 召回，也比单纯拆关键词更不容易过宽。

### CitationRecommender

`CitationRecommender` 负责统一编排候选来源和证据验证：

```text
claim profile
-> current_bib metadata recall
-> zotero metadata recall
-> dedupe
-> metadata rerank
-> top-N evidence verification
-> verdict rerank
```

去重不只看 citation key。由于 Zotero 里同一篇论文可能被导入多次，并由 Better BibTeX 生成不同的 key，V2.5 会按以下任一条件合并重复候选：

- DOI 规范化后一致；
- 标题规范化后一致；
- citation key 一致。

合并时优先保留当前 `.bib` 条目，其次保留有 citation key、有 PDF 路径、置信度更高的条目。

默认最大返回 5 条候选，可通过 `citation-agent.maxSuggestions` 配置。证据验证默认参数：

```text
maxMetadataCandidates = 50
maxAbstractCandidates = 8
maxEvidencePassages = 5
```

### EvidenceRetriever / EvidenceJudge

`EvidenceRetriever` 读取短 passage：

- `.bib` abstract；
- Zotero `abstractNote`；
- Zotero PDF 附件路径指向的本地 PDF。

PDF passage 抽取会过滤明显的“外部论文转述”片段。例如在 GRAND-SLAM 论文中出现的 `MAGIC-SLAM [38] extends ...` 会被识别为 GRAND-SLAM 对其他工作的引用描述，而不是 GRAND-SLAM 自己的主张证据。LLM judge 的系统提示也要求只使用候选论文自己的 abstract、method、finding 或 contribution 证据。

PDF passage 还会进行 evidence 质量过滤，丢弃不适合送给 LLM 判断的片段：

- 参考文献列表条目，例如包含多个作者缩写、年份、会议/期刊名、arXiv ID 的 citation list；
- 图表 caption 或架构图文字，例如 `AGENT SIDE ... SERVER SIDE ... Fig. 3`；
- OCR/图中文字序列，例如大量短词、全大写词、缺少句子结构的组件列表；
- 过短或没有基本句子结构的 PDF 文本块。

该过滤对 PDF 更严格；对 abstract/metadata 更保守，只过滤明显参考文献或图表噪声，避免误伤较短摘要。

PDF 读取依赖 `PyMuPDF/fitz`。如果当前虚拟环境没有安装该包，系统会记录日志并回退到 abstract/metadata，不会中断推荐流程。

`EvidenceJudge` 判断 passage 是否支持 claim：

- 有模型 API：调用 LiteLLM，让 LLM 阅读最多 40 个短 passage 中与某篇候选相关的部分，并返回 JSON verdict。
- 无模型 API 或调用失败：使用本地规则，根据研究对象短语和谓词短语覆盖程度判断。

候选 `confidence` 表示“作为引用候选的支持置信度”，不是 LLM 对某个负面判断的自信度。因此 verdict 会限制 confidence 上限：`insufficient_evidence` 最高 35%，`contradiction` 最高 25%，避免出现“证据不足但 100%”的误导性展示。

为了提高可复现性，V2.5 对候选和证据 passage 做确定性排序：

- Zotero 返回条目先按规范化标题、年份、item key 排序，再进入候选池；
- 元数据候选同分时按来源、规范化标题、citation key 排序；
- 证据验证后同分时按 verdict、支持置信度、是否有 citation key、来源、标题排序；
- evidence passage 同分时按分数、来源、页码、section、文本内容排序；
- LLM 返回的 `evidence_index` 只有在它与确定性最高分 passage 打平时才会作为主证据，否则 UI 使用确定性最高分 passage。

这不能消除远端 LLM 服务本身的全部非确定性，但能避免 Zotero 顺序、PDF 抽取顺序、同分排序导致的候选列表和展示证据随机漂移。

verdict 含义：

```text
direct_support        passage 明确支持选中文段核心含义
partial_support       passage 支持部分含义，但不完整
related_only          主题相关，但没有表达同一观点
contradiction         passage 与 claim 冲突
insufficient_evidence 没找到可用 passage
metadata_match        仅用于未启用证据验证的兼容状态
```

### ZoteroClient

V2 的 Zotero 是可选只读适配器，支持两种 endpoint：

1. Zotero 官方 Local API：

```text
http://127.0.0.1:23119/api
```

启用后会按多条 claim-aware 查询访问：

```text
GET /api/users/0/items/top?q=...&qmode=everything&limit=...&format=json
```

如果 Better BibTeX 已安装并可访问，V2 会继续调用：

```text
POST /better-bibtex/json-rpc
method: item.citationkey
params: [[zoteroItemKey1, zoteroItemKey2, ...]]
```

这一步用于按需获取候选 Zotero 条目的 citation key。它不要求用户提前手动给全部 Zotero 条目生成 key。

2. mock/bridge endpoint：

```text
GET {zoteroEndpoint}/search?q=...&limit=...
GET {zoteroEndpoint}/items?q=...&limit=...
```

期望返回 JSON list 或 `{ "items": [...] }`。item 至少需要 `title`，可选：

```text
citationKey
creators / authors
date / year
publicationTitle / conferenceName
DOI
url
pdfPath
abstractNote
```

如果 Zotero 不在线或 endpoint 不可用，V2 会静默回退到当前 `.bib` 搜索。

## 前端设计

`EasyCite: 检查选中内容` 的结果弹窗现在会显示候选数量。

如果有候选，检查完成后会自动打开一个持久的编辑器面板：

```text
EasyCite Candidates
```

这个面板是候选审核的主界面，不会像右下角通知一样自动消失。即使切到完整证据页、LaTeX 文件或其他标签页，候选面板仍会保留在编辑器标签栏里。

如果手动关闭了面板，候选审核仍会保留在状态栏：

```text
EasyCite: 返回候选
```

点击状态栏即可恢复候选面板。也可以通过命令面板执行：

```text
EasyCite: 返回候选审核
```

候选面板展示：

- 标题
- 年份
- 来源：`current_bib` 或 `zotero`
- citation key
- evidence verdict
- 置信度
- 作者、venue、证据判断原因
- 一条压缩后的 `最相关文段` 预览

用户可以：

- `查看完整证据`：在旁边打开 Markdown 详情页，展开待引用文段、候选元数据、判断理由、最相关文段和全部备选 evidence passages。候选面板会继续保留，切回 `EasyCite Candidates` 标签即可继续比较其他候选。
  - 详情页会保留英文原文。
  - 如果已通过 `EasyCite: 配置模型服务` 配置模型 API，会按需生成中文对照。
  - 如果未配置模型 API，详情页仍可打开，并显示“未配置模型 API，暂不生成中文对照”。
  - 中文对照使用独立的翻译超时，默认 60 秒。翻译请求按小批量执行；如果批量超时，会自动拆成单段重试，能返回已成功翻译的部分，而不是整页全部失败。
- `加入引用列表`：把该候选加入待插入集合，状态栏会显示已选数量。
- `仅插入此条`：只插入当前候选。
- `插入已选择的 N 条引用`：候选列表顶部操作，将多条候选一次性插入为同一个 `\cite{key1,key2,...}`。

面板底部可能出现：

```text
未入选但已识别的候选
```

这里展示已被召回或验证、但没有进入最终推荐前五的诊断候选。它们不会默认作为推荐插入，但可以打开完整证据，用来判断某篇文章是“没被 Zotero 召回”，还是“被召回后因证据不足/置信度不足没有上榜”。

插件插入格式为：

```latex
~\cite{citation_key}
```

## 新增配置

```json
{
  "citation-agent.maxSuggestions": 5,
  "citation-agent.maxMetadataCandidates": 50,
  "citation-agent.maxAbstractCandidates": 8,
  "citation-agent.maxEvidencePassages": 5,
  "citation-agent.enableEvidenceVerification": true,
  "citation-agent.zoteroEnabled": false,
  "citation-agent.zoteroEndpoint": ""
}
```

当前 `.bib` 搜索不依赖 Zotero。Zotero 和证据验证都可以独立关闭。

翻译超时可通过环境变量调整：

```bash
export CITATION_AGENT_TRANSLATION_TIMEOUT_SECONDS=90
```

该设置只影响完整证据页的中文对照生成，不改变 evidence judge 的默认模型超时。

## 验证方式

```bash
cd <repository-root>
source "$HOME/.nvm/nvm.sh"
npm run compile
.venv/bin/python -m pytest src/test/python_tests/test_audit_claim.py src/test/python_tests/test_get_cwd.py
.venv/bin/python -m pytest src/test/python_tests
.venv/bin/python -m pip check
```

当前验证结果：

```text
webpack compiled successfully
36 passed
No broken requirements found.
```

手动验证：

1. 在 VS Code 打开 `<repository-root>`。
2. 运行 `Debug EasyCite v2`。
3. 在扩展开发主机中打开一个 `.tex` 文件，并确保同目录或工作区内有 `.bib`。
4. 选中一句需要引用的话。
5. 执行 `EasyCite: 检查选中内容`。
6. 如果找到候选，点击 `查看候选引用`。
7. 选择候选并确认插入。

## 手动体验 Zotero 候选

V2 可以直接读取 Zotero 官方 Local API。先在 Zotero 桌面端开启：

```text
设置 -> 高级 -> 允许本机其他应用程序与 Zotero 通信
```

然后确认 Zotero 正在运行。在 WSL 终端里测试：

```bash
curl 'http://127.0.0.1:23119/api/users/0/items/top?limit=3&format=json' \
  -H 'Zotero-API-Version: 3'
```

如果能返回 JSON，VS Code 中配置：

```json
{
  "citation-agent.zoteroEnabled": true,
  "citation-agent.zoteroEndpoint": "http://127.0.0.1:23119/api",
  "citation-agent.maxSuggestions": 5
}
```

如果 Zotero 跑在 Windows，而 VS Code/Python 后端跑在 WSL，`127.0.0.1` 可能连不到 Windows Zotero。此时用 Windows host IP，例如：

```bash
grep nameserver /etc/resolv.conf
```

然后配置：

```json
{
  "citation-agent.zoteroEndpoint": "http://<Windows-host-ip>:23119/api"
}
```

为了不依赖真实 Zotero，也可以用 mock server 体验完整流程。

在一个终端启动 mock Zotero：

```bash
cd <repository-root>
.venv/bin/python scripts/mock_zotero_server.py
```

预期输出：

```text
[mock-zotero] running http://127.0.0.1:8765
[mock-zotero] supports GET /search?q=...&limit=...
```

在 VS Code 设置中开启：

```json
{
  "citation-agent.zoteroEnabled": true,
  "citation-agent.zoteroEndpoint": "http://127.0.0.1:8765",
  "citation-agent.maxSuggestions": 5
}
```

然后重新启动 `Debug EasyCite v2`。在扩展开发主机里选中需要引用的句子并执行：

```text
EasyCite: 检查选中内容
```

如果当前 `.bib` 和 mock Zotero 都命中，点击 `查看候选引用` 后应看到来源分别为：

```text
current_bib
zotero
```

mock server 终端会打印插件发出的请求，例如：

```text
[mock-zotero] /search q='Existing multi-agent Gaussian SLAM...' limit=4
```

如果使用真实 Zotero + Better BibTeX，`输出 -> EasyCite` 中应能看到：

```text
Claim-aware retrieval profile: phrases=..., queries=...
Zotero Local API query 'Gaussian SLAM' returned ... raw item(s).
Zotero Local API returned ... raw unique item(s), ... citable item(s)
Requesting Better BibTeX JSON-RPC method: item.citationkey
Better BibTeX HTTP response: 200
Better BibTeX citation key lookup returned ... key(s)
Evidence verdict: direct_support confidence=...
```

## V2 边界

V2.5 仍不做：

- PaperQA2 / 向量数据库级别的长期索引；
- 自动 OCR 扫描版 PDF；
- 自动更新 `.bib`；
- 在线学术数据库检索；
- 实时诊断；
- 侧边栏/Webview 审核页面。

V2.5 的证据验证是“短 passage 级别”的轻量实现。它已经能避免纯元数据匹配带来的大量误判，但还不是完整的论文理解系统。后续 V3 可以把 passage 检索升级为嵌入索引、PDF 分块缓存和更强的 LLM citation verifier。
