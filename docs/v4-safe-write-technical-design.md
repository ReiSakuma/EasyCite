# EasyCite V4 安全写入闭环技术设计

## 目标

V4 在 V3 的证据驱动候选推荐基础上，推进到更完整的写作闭环：

```text
候选证据审核
-> 用户选择一条或多条引用
-> 定位当前 TeX 工程使用的 .bib 文件
-> 检查 citation key 是否已经存在
-> 对缺失的 Zotero 条目调用 Better BibTeX 导出 BibTeX
-> 展示写入计划
-> 用户确认后同时修改 .tex 和 .bib
```

V4 的重点不是扩大检索来源，而是让已经确认的引用能安全、完整地写入论文工程。

## 已实现能力

### 1. 从 V3 复制独立版本

代码路径：

```text
<repository-root>
```

V4 不修改 V3，后续调试和验证均在 v4 路径下进行。

### 2. BibTeX 补全协议

新增 LSP 请求：

```text
citationAgent/resolveBibtex
```

前端发送：

```json
{
  "citation_keys": ["smith2024slam"],
  "suggestions": [...]
}
```

后端返回：

```json
{
  "entries": [
    {
      "citation_key": "smith2024slam",
      "bibtex": "@article{smith2024slam,...}",
      "source": "zotero"
    }
  ],
  "missing": []
}
```

相关文件：

```text
src/types/protocol.ts
bundled/tool/lsp_server.py
bundled/tool/citation_agent/bibtex_resolver.py
```

### 3. Better BibTeX 导出

V4 使用 Better BibTeX JSON-RPC：

```text
POST /better-bibtex/json-rpc
method: item.export
params: [[citation_keys], "Better BibTeX"]
```

这会按照 Zotero/BBT 当前配置导出 BibTeX 条目，并保留 BBT citation key。

相关文件：

```text
bundled/tool/citation_agent/zotero.py
```

### 4. 最小 TeX 工程 BibTeX 定位

前端在应用引用前尝试识别当前工程使用的 `.bib` 文件：

```latex
\addbibresource{references.bib}
\bibliography{references}
```

如果当前 TeX 文件没有声明，则在工作区内搜索 `.bib` 文件，并优先选择：

```text
当前文件同目录下的 references.bib / reference.bib / refs.bib / main.bib
```

如果无法识别 `.bib` 文件，V4 不会盲目创建或写错文件，只允许用户选择“仅插入 cite”。

### 5. 写入计划确认

用户点击“仅插入此条”或“插入已选择引用”后，不再直接写入，而是先显示计划：

```text
TeX: 将插入 \cite{...}
BibTeX: 将修改哪个 .bib
已存在 key 数量
待追加 BibTeX 条目数量
无法补全的 key 和原因
```

用户可以选择：

```text
插入引用并更新 BibTeX
仅插入 cite
取消
```

### 6. 引用插入规则增强

V4 对插入位置做了更系统的处理：

```text
英文标点前插入：claim\cite{key}.
中文标点前插入：claim\cite{key}。
已有 \cite{a} 时合并为 \cite{a,b}
多候选插入为 \cite{a,b,c}
重复 key 自动去重
```

相关文件：

```text
src/commands/auditSelection.ts
```

## 当前边界

V4 暂未实现：

```text
latexmk 编译验证
修改 diff 可视化页面
完整 TeX 主文件解析
跨 \input / \include 的 bibliography 定位
联网学术检索
从非 Zotero 来源自动解析 BibTeX
Zotero 写入、导入、分类整理
```

这些应留给 V4 后续小版本或 V5。

## 测试

TypeScript 编译：

```bash
npm run compile
```

Python 单元测试：

```bash
.venv/bin/python -m pytest src/test/python_tests
```

当前验证结果：

```text
npm run compile: pass
pytest src/test/python_tests: 54 passed
```

## 手动测试建议

1. 用 VS Code 打开 `<repository-root>`。
2. 启动扩展开发宿主。
3. 在扩展开发宿主中打开论文工作区。
4. 确保 `test.tex` 包含：

```latex
\addbibresource{references.bib}
```

5. 确保 Zotero 和 Better BibTeX 可用。
6. 运行：

```text
EasyCite: 检查选中内容
```

7. 在候选面板中选择 Zotero 候选。
8. 点击：

```text
加入引用列表
插入已选择的 N 条引用
```

9. 预期弹出写入计划，确认后：

```text
.tex 中插入 \cite{...}
references.bib 中自动追加缺失 BibTeX 条目
```
