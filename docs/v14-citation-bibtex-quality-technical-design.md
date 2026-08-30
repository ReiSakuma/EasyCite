# EasyCite v14 Citation / BibTeX 质量改造

## 目标

v14 聚焦引用落盘质量。前几个版本解决了“找什么文献更可信”，这一版解决“用户点插入后，TeX 和 BibTeX 是否真的处于可编译、可追溯、可审计的状态”。

科研写作里，错误的 `\cite{}` 或损坏的 `.bib` 条目会在论文编译、投稿检查和后续协作中造成非常隐蔽的问题。因此 v14 不只插入 citation key，而是引入写入前预检、来源可信度标注和写入后校验。

## BibTeX 来源可信度

v14 定义 BibTeX provenance contract：

1. `better-bibtex:zotero`
   - 高可信。
   - 来自 Zotero + Better BibTeX 的标准导出。
2. `zotero-metadata:*`
   - 中可信。
   - 来自 Zotero metadata 拼装。
3. `doi-metadata:*` / `arxiv-metadata:*`
   - 中可信。
   - 来自可验证外部元数据。
4. `candidate-metadata-fallback:*`
   - 低可信。
   - 由候选文献已有元数据兜底生成，必须人工核对。

后端 `resolve_bibtex_entries` 会返回：

- `source`
- `source_trust`
- `source_rank`
- `requires_user_verification`
- `quality_warnings`

前端写入计划面板会用这些字段展示来源可信度。

## 写入前预检

写入前检查：

- 是否有 citation key。
- 是否识别到可写 `.bib` 文件。
- 目标 `.bib` 是否可写或可创建。
- 当前 TeX bibliography 声明是否指向目标 `.bib`。
- 缺失 key 是否能补全 BibTeX。
- 待追加 BibTeX 的 entry key 是否与 citation key 一致。
- 待追加 BibTeX 的大括号是否平衡。
- 是否使用低可信 fallback。

如果存在 error，默认禁用“插入引用并更新 BibTeX”，用户只能选择“仅插入 cite”或取消。

## 写入后校验

写入完成后，v14 会再次读取实际文件，验证：

- TeX 中是否真的出现插入的 citation key。
- bibliography 声明是否真的指向目标 `.bib`。
- 目标 `.bib` 中是否包含全部 citation key。
- `.bib` 中是否存在重复 key。
- `.bib` 是否存在无法解析或括号不平衡的条目。

这一步用于捕捉 VS Code WorkspaceEdit 被部分拒绝、文件外部修改、BibTeX 条目损坏等情况。

## 用户端变化

用户点击“插入引用并更新 BibTeX”时，会看到更清晰的 Apply Plan：

- BibTeX 来源可信度。
- 低可信 fallback 的警告。
- 写入前预检列表。
- 待追加 BibTeX 原文。

写入后如果发现问题，会弹出校验报告；如果校验全部通过，则安静完成并在日志中记录通过信息。

## 验证方式

1. 使用 Zotero + Better BibTeX 候选：
   - 预期来源显示高可信。
   - 写入后 `.bib` 有对应 key。
2. 关闭 Better BibTeX 或使用非 Zotero 元数据候选：
   - 预期来源显示低可信 fallback。
   - UI 提醒提交前必须人工核对。
3. 手动在 `.bib` 中制造重复 key：
   - 写入后校验应报告重复 key。
4. 手动破坏一个 BibTeX 大括号：
   - 写入后校验应报告结构问题。
5. 没有 `.bib` 文件的 TeX 项目：
   - 系统应创建默认 `references.bib`，并同步写入 `\addbibresource{references.bib}`。

## 面试表达

v14 可以作为 Tool Reliability 的落地案例：

> 我把“插入引用”从简单文本编辑升级成一个可审计的 write transaction：先生成 plan，标注 BibTeX 来源可信度，执行前做 schema 和文件权限校验，执行后重新读取真实文件做 post-apply verification。这样用户不是只相信 Agent 说“我写入了”，而是能看到 TeX 和 BibTeX 状态确实一致。
