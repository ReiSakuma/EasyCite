# EasyCite v12: Tool Registry / Health Check

v12 的目标是把 EasyCite 从“出错后看日志排查”推进到“运行前能看到工具链状态”的成品化阶段。

## 用户端新增功能

新增命令：

```text
EasyCite: 打开工具健康检查
```

该面板会只读检测：

- Python runtime
- 必需 Python 包：pydantic、pygls、lsprotocol、langgraph
- 模型 API 配置与 litellm
- Zotero Local API
- Better BibTeX JSON-RPC
- embedding 模型与 fastembed/hash provider
- PDF evidence reader / PyMuPDF
- 当前工作区 BibTeX 文件
- 项目反馈记忆数据库

每个工具有四种状态：

```text
ok        可用
warning   可降级但建议修复
error     关键能力不可用
disabled  用户未启用，不一定是错误
```

## 设计原则

1. 健康检查默认只读，不自动安装依赖、不修改配置、不启动 Zotero。
2. 修复动作复用已有命令：设置向导、配置模型服务、配置 Zotero 文献范围、清空项目记忆。
3. 面板直接展示细节与建议，减少用户从 Output 日志里找错误的成本。
4. 后端 health schema 与推荐 workflow 解耦，后续可逐步演化成正式 Tool Registry。

## 后端实现

新增：

```text
bundled/tool/citation_agent/tool_health.py
```

核心入口：

```python
run_tool_health_check(params, logger=None)
```

通过 LSP custom request 暴露：

```text
citationAgent/toolHealth
```

## 前端实现

新增：

```text
src/commands/toolHealth.ts
```

该文件负责：

- 确保 Python 后端运行
- 发送 `ToolHealthRequest`
- 渲染 VS Code webview
- 提供“重新检查 / 设置向导 / 配置模型 / 配置 Zotero 范围 / 清空项目记忆”按钮

## 面试表达

v12 可以这样讲：

> EasyCite 依赖 Zotero、Better BibTeX、LLM、embedding、PDF parser 和 BibTeX 文件系统。为了避免用户只看到“候选为空”却不知道哪里坏了，我把这些外部能力抽象成 health-check item，并用统一状态展示可用性、降级路径和修复建议。后续这层可以继续演化成 Tool Registry，统一 timeout、retry、schema validation 和 fallback policy。
