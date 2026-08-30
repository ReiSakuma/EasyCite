# Security Policy

## Reporting a Vulnerability

请不要通过公开 GitHub Issue 报告安全漏洞。请使用 GitHub 仓库的 Private Vulnerability Reporting 功能联系维护者。

报告中建议包含：

- 受影响的 EasyCite 版本
- 漏洞类型和影响范围
- 最小复现步骤或测试工程
- 涉及的文件、命令或接口
- 建议的修复方向

请勿发送真实 API Key、私人 Zotero 数据或未公开论文。EasyCite 的模型 API Key 使用 VS Code SecretStorage 保存，运行记录只应保存 `set`/`missing` 状态，不应包含密钥本身。
