import * as vscode from 'vscode';
import { State } from 'vscode-languageclient';
import { LanguageClient } from 'vscode-languageclient/node';
import { traceError } from '../common/log/logging';
import { ToolHealthItem, ToolHealthRequest, ToolHealthResult, ToolHealthStatus } from '../types/protocol';

const TOOL_HEALTH_TIMEOUT_MS = 45_000;

type ClientProvider = () => LanguageClient | undefined;

interface ToolHealthMessage {
    command?: 'refresh' | 'setupWizard' | 'configureProvider' | 'configureZoteroScope' | 'clearMemory';
}

function withTimeout<T>(promise: Thenable<T>, timeoutMs: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => {
            reject(new Error(`EasyCite request timed out after ${timeoutMs}ms.`));
        }, timeoutMs);

        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}

export function registerToolHealthCommand(
    context: vscode.ExtensionContext,
    getClient: ClientProvider,
    ensureServer: () => Promise<LanguageClient | undefined>,
): void {
    let panel: vscode.WebviewPanel | undefined;
    let latestResult: ToolHealthResult | undefined;

    async function getRunningClient(): Promise<LanguageClient | undefined> {
        let client = getClient();
        if (client?.state === State.Running) {
            return client;
        }
        client = await ensureServer();
        return client?.state === State.Running ? client : undefined;
    }

    async function runHealthCheck(): Promise<ToolHealthResult | undefined> {
        const client = await getRunningClient();
        if (!client) {
            vscode.window.showErrorMessage('EasyCite: Python 后端尚未启动，无法运行工具健康检查。');
            return undefined;
        }

        const activeDocument = vscode.window.activeTextEditor?.document;
        const workspacePath = activeDocument
            ? vscode.workspace.getWorkspaceFolder(activeDocument.uri)?.uri.fsPath
            : vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        const filePath = activeDocument?.uri.scheme === 'file' ? activeDocument.uri.fsPath : '';

        try {
            return await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: 'EasyCite: 正在检查工具链状态',
                    cancellable: false,
                },
                async () =>
                    withTimeout(
                        client.sendRequest(ToolHealthRequest, {
                            workspace_path: workspacePath ?? '',
                            file_path: filePath,
                        }),
                        TOOL_HEALTH_TIMEOUT_MS,
                    ),
            );
        } catch (error) {
            traceError(`EasyCite tool health check failed: ${error}`);
            vscode.window.showErrorMessage(`EasyCite: 工具健康检查失败：${error}`);
            return undefined;
        }
    }

    async function openOrRefresh(): Promise<void> {
        const result = await runHealthCheck();
        if (!result) {
            return;
        }
        latestResult = result;
        if (!panel) {
            panel = vscode.window.createWebviewPanel(
                'citationAgentToolHealth',
                'EasyCite Tool Health',
                vscode.ViewColumn.Beside,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true,
                },
            );
            panel.onDidDispose(
                () => {
                    panel = undefined;
                },
                undefined,
                context.subscriptions,
            );
            panel.webview.onDidReceiveMessage(
                async (message: ToolHealthMessage) => {
                    if (message.command === 'refresh') {
                        await openOrRefresh();
                        return;
                    }
                    if (message.command === 'setupWizard') {
                        await vscode.commands.executeCommand('citation-agent.setupWizard');
                        await openOrRefresh();
                        return;
                    }
                    if (message.command === 'configureProvider') {
                        await vscode.commands.executeCommand('citation-agent.configureProvider');
                        await openOrRefresh();
                        return;
                    }
                    if (message.command === 'configureZoteroScope') {
                        await vscode.commands.executeCommand('citation-agent.configureZoteroScope');
                        await openOrRefresh();
                        return;
                    }
                    if (message.command === 'clearMemory') {
                        await vscode.commands.executeCommand('citation-agent.clearProjectMemory');
                        await openOrRefresh();
                    }
                },
                undefined,
                context.subscriptions,
            );
        }
        panel.title = `EasyCite Tool Health: ${result.summary.overall_status.toUpperCase()}`;
        panel.webview.html = renderToolHealthHtml(panel.webview, result);
        panel.reveal(vscode.ViewColumn.Beside);
    }

    context.subscriptions.push(vscode.commands.registerCommand('citation-agent.openToolHealth', openOrRefresh));

    void latestResult;
}

function renderToolHealthHtml(webview: vscode.Webview, result: ToolHealthResult): string {
    const nonce = Date.now().toString(36);
    const summaryClass = statusClass(result.summary.overall_status);
    const itemCards = result.items.map(renderToolHealthItem).join('');
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EasyCite Tool Health</title>
    <style>
        body {
            background: var(--vscode-editor-background);
            color: var(--vscode-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            margin: 0;
            padding: 18px;
        }
        h1 {
            font-size: 20px;
            margin: 0 0 10px;
        }
        h2 {
            font-size: 15px;
            margin: 0;
        }
        p {
            line-height: 1.5;
            margin: 6px 0;
        }
        button {
            background: var(--vscode-button-background);
            border: 0;
            border-radius: 3px;
            color: var(--vscode-button-foreground);
            cursor: pointer;
            padding: 5px 10px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .summary {
            border-left: 4px solid var(--vscode-textLink-foreground);
            margin: 12px 0 16px;
            padding: 10px 12px;
            background: var(--vscode-editorWidget-background);
        }
        .summary.error {
            border-left-color: var(--vscode-errorForeground);
        }
        .summary.warning {
            border-left-color: var(--vscode-editorWarning-foreground);
        }
        .summary.ok {
            border-left-color: var(--vscode-testing-iconPassed);
        }
        .toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 12px 0;
        }
        .grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }
        .card {
            border: 1px solid var(--vscode-panel-border);
            border-radius: 6px;
            padding: 12px;
        }
        .card.error {
            border-color: var(--vscode-errorForeground);
        }
        .card.warning {
            border-color: var(--vscode-editorWarning-foreground);
        }
        .card.ok {
            border-color: var(--vscode-testing-iconPassed);
        }
        .card.disabled {
            opacity: 0.76;
        }
        .head {
            align-items: center;
            display: flex;
            gap: 8px;
            justify-content: space-between;
        }
        .badge {
            border-radius: 3px;
            font-size: 11px;
            padding: 2px 6px;
            text-transform: uppercase;
        }
        .badge.error {
            background: var(--vscode-errorForeground);
            color: var(--vscode-editor-background);
        }
        .badge.warning {
            background: var(--vscode-editorWarning-foreground);
            color: var(--vscode-editor-background);
        }
        .badge.ok {
            background: var(--vscode-testing-iconPassed);
            color: var(--vscode-editor-background);
        }
        .badge.disabled {
            background: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
        }
        .muted,
        .meta {
            color: var(--vscode-descriptionForeground);
        }
        ul {
            margin: 8px 0 0 20px;
            padding: 0;
        }
        li {
            line-height: 1.45;
            margin: 3px 0;
            overflow-wrap: anywhere;
        }
        code {
            overflow-wrap: anywhere;
        }
    </style>
</head>
<body>
    <h1>EasyCite Tool Health</h1>
    <section class="summary ${summaryClass}">
        <strong>${htmlEscape(result.summary.message)}</strong>
        <p class="meta">生成时间：${htmlEscape(result.generated_at)} · workspace: <code>${htmlEscape(result.workspace_path || '<none>')}</code></p>
        <p>OK ${result.summary.ok} · Warning ${result.summary.warning} · Error ${result.summary.error} · Disabled ${result.summary.disabled}</p>
    </section>
    <div class="toolbar">
        <button data-command="refresh">重新检查</button>
        <button data-command="setupWizard">打开设置向导</button>
        <button data-command="configureProvider">配置模型服务</button>
        <button data-command="configureZoteroScope">配置 Zotero 文献范围</button>
        <button data-command="clearMemory">清空项目记忆</button>
    </div>
    <main class="grid">${itemCards}</main>
    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        document.addEventListener('click', (event) => {
            const button = event.target.closest('button[data-command]');
            if (!button || button.disabled) {
                return;
            }
            vscode.postMessage({ command: button.getAttribute('data-command') });
        });
    </script>
</body>
</html>`;
}

function renderToolHealthItem(item: ToolHealthItem): string {
    const cls = statusClass(item.status);
    const latency = typeof item.latency_ms === 'number' ? ` · ${item.latency_ms.toFixed(0)} ms` : '';
    const details =
        item.details.length > 0
            ? `<ul>${item.details.map((detail) => `<li>${htmlEscape(detail)}</li>`).join('')}</ul>`
            : '<p class="muted">无额外细节。</p>';
    const actions =
        item.actions.length > 0
            ? `<ul>${item.actions.map((action) => `<li>${htmlEscape(action)}</li>`).join('')}</ul>`
            : '<p class="muted">无需操作。</p>';
    return `
        <article class="card ${cls}">
            <div class="head">
                <h2>${htmlEscape(item.label)}</h2>
                <span class="badge ${cls}">${htmlEscape(item.status)}</span>
            </div>
            <p>${htmlEscape(item.summary)}<span class="meta">${latency}</span></p>
            <details>
                <summary>检查细节</summary>
                ${details}
            </details>
            <details>
                <summary>建议操作</summary>
                ${actions}
            </details>
        </article>
    `;
}

function statusClass(status: ToolHealthStatus): string {
    if (status === 'error' || status === 'warning' || status === 'disabled') {
        return status;
    }
    return 'ok';
}

function htmlEscape(value: string): string {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
