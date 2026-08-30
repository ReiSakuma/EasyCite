# EasyCite v1 Technical Design

## 目标

V1 在 V0 通信骨架上增加“单句引用需求检测”：

```text
LaTeX 选中文本
-> VS Code 命令
-> Python 后端
-> 判断是否需要引用
-> 返回 citation_needed / reason / confidence / claim_type
```

V1 仍然不做 Zotero、BibTeX 搜索、PaperQA2、PDF 解析、在线学术检索、Webview 或 TeX/BibTeX 修改。

## 新增能力

- 接入 LiteLLM 作为统一模型网关。
- 新增模型配置命令 `EasyCite: 配置模型服务`。
- API Key 使用 VS Code `SecretStorage` 保存，不写入项目文件、`.env` 或 `settings.json`。
- 后端输出结构化引用需求判断。
- 无 API Key 或模型调用失败时，回退本地启发式规则。

## 前端设计

### 命令

V1 提供两个主要命令：

```text
EasyCite: 检查选中内容
EasyCite: 配置模型服务
```

`配置模型服务` 会收集：

- provider: OpenAI / DeepSeek / OpenRouter / OpenAI-Compatible
- model: LiteLLM 模型名
- apiBase: 可选
- apiKey: SecretStorage

插件启动或重启 Python 后端前，会把非敏感配置和 SecretStorage 中的 API Key 注入进程环境变量：

```text
CITATION_AGENT_USE_LLM
CITATION_AGENT_PROVIDER
CITATION_AGENT_MODEL
CITATION_AGENT_API_BASE
CITATION_AGENT_API_KEY
```

### 请求协议

仍然使用 LSP custom request：

```text
citationAgent/auditClaim
```

前端使用 `RequestType(..., ParameterStructures.byName)`，避免 JSON-RPC 把参数转成 positional array。

## 后端设计

新增模块：

```text
bundled/tool/citation_agent/
├── __init__.py
├── config.py
├── detection.py
└── schemas.py
```

### CitationNeedDetector

检测顺序：

1. 如果 claim 为空，返回错误。
2. 如果已有 `\cite{}`，返回 `citation_needed: false`。
3. 如果 `use_llm=true` 且有 API Key，调用 LiteLLM。
4. 如果 LLM 失败，回退本地规则，并在 reason 中说明回退。
5. 如果未启用 LLM，直接使用本地规则。

### 本地规则

V1 启发式规则识别：

- `related_work_summary`: existing / prior / generally / commonly / rely on 等领域概括。
- `empirical_result`: outperform / improve / percent / times 等实验或数值结果。
- `definition`: is defined as / refers to / known as 等定义句。
- `writing_only`: 短写作性文本。
- `other`: 未识别为强引用需求。

### LLM 输出

LLM 必须返回 JSON：

```json
{
  "citation_needed": true,
  "reason": "This sentence summarizes prior work.",
  "confidence": 0.86,
  "claim_type": "related_work_summary"
}
```

后端使用 Pydantic 校验，并补充：

```json
{
  "source": "llm",
  "suggestions": [],
  "recommended_rewrite": null
}
```

## 输出协议

V1 response 新增：

```text
claim_type
source
```

`source` 可为：

```text
llm
heuristic
```

完整 schema 见：

```text
protocol/audit-claim.schema.json
```

## 验证方式

无 API Key 也可以验证 V1：

```bash
cd <repository-root>
source "$HOME/.nvm/nvm.sh"
npm run compile
.venv/bin/python -m pytest src/test/python_tests/test_audit_claim.py src/test/python_tests/test_get_cwd.py
```

预期：

```text
webpack compiled successfully
23 passed
```

手动验证：

1. 在 VS Code 打开 `<repository-root>`。
2. 运行 `Debug EasyCite v1`。
3. 在扩展开发主机中打开 `.tex` 文件。
4. 选中：

```latex
Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.
```

5. 执行 `EasyCite: 检查选中内容`。
6. 预期弹窗包含：

```text
需要引用，related_work_summary，heuristic
```

配置模型后，预期 `source` 变为：

```text
llm
```

## V1 边界

V1 不推荐具体论文，不生成 BibTeX，不插入 `\cite{}`。

它只回答：

```text
这句话是否需要引用？
为什么？
置信度是多少？
这是什么类型的科研陈述？
结果来自 LLM 还是本地规则？
```
