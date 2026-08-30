import * as path from 'path';
import * as vscode from 'vscode';
import { API_KEY_SECRET, configureModelProvider } from './configureProvider';
import { traceInfo } from '../common/log/logging';
import {
    checkPythonRuntime,
    ensureRequiredPythonRuntime,
    initializePrivatePythonLibEnvironment,
    installOptionalPackageGroup,
} from '../common/runtimeEnvironment';
import { ensureZoteroAvailable, isBetterBibTeXAvailable, isZoteroAvailable } from '../common/zotero';

const SETUP_COMPLETED_KEY = 'citation-agent.setupCompleted.v17';
const SETUP_DISMISSED_KEY = 'citation-agent.setupPromptDismissed.v17';

interface WizardChoice extends vscode.QuickPickItem {
    action: string;
}

export function registerSetupWizardCommand(
    context: vscode.ExtensionContext,
    namespace: string,
    restartServer: () => Promise<void>,
): void {
    initializePrivatePythonLibEnvironment(context);
    context.subscriptions.push(
        vscode.commands.registerCommand('citation-agent.setupWizard', async () => {
            await runSetupWizard(context, namespace, restartServer);
        }),
    );
}

export async function maybeShowFirstRunSetup(
    context: vscode.ExtensionContext,
    namespace: string,
    restartServer: () => Promise<void>,
): Promise<void> {
    initializePrivatePythonLibEnvironment(context);
    if (context.globalState.get<boolean>(SETUP_COMPLETED_KEY, false)) {
        return;
    }
    if (context.globalState.get<boolean>(SETUP_DISMISSED_KEY, false)) {
        return;
    }
    const action = await vscode.window.showInformationMessage(
        'EasyCite: 首次使用建议运行设置向导，检查 Python、Zotero、Better BibTeX、模型 API、embedding 和 PDF 读取能力。',
        '开始设置',
        '稍后',
    );
    if (action === '开始设置') {
        await runSetupWizard(context, namespace, restartServer);
    } else {
        await context.globalState.update(SETUP_DISMISSED_KEY, true);
    }
}

export async function runSetupWizard(
    context: vscode.ExtensionContext,
    namespace: string,
    restartServer: () => Promise<void>,
): Promise<void> {
    const started = await vscode.window.showQuickPick<WizardChoice>(
        [
            {
                label: '完整设置',
                description: '检查 Python、Zotero/Better BibTeX、模型 API、embedding 和 PDF 依赖',
                action: 'full',
            },
            {
                label: '仅检查 Python 依赖',
                description: '修复后端启动所需依赖',
                action: 'python',
            },
            {
                label: '取消',
                action: 'cancel',
            },
        ],
        {
            title: 'EasyCite: 设置向导',
            placeHolder: '选择要配置的项目',
            ignoreFocusOut: true,
        },
    );
    if (!started || started.action === 'cancel') {
        return;
    }

    await configureDependencyInstallMode();
    const runtimeReady = await ensureRequiredPythonRuntime(context, namespace);
    if (!runtimeReady) {
        return;
    }
    await showPythonRuntimeSummary(context, namespace);

    if (started.action === 'python') {
        await context.globalState.update(SETUP_COMPLETED_KEY, true);
        await context.globalState.update(SETUP_DISMISSED_KEY, true);
        return;
    }

    await configureZoteroAndBetterBibTeX();
    await configureModelApi(context, namespace, restartServer);
    await configureEmbedding(context, namespace);
    await configurePdfReader(context, namespace);

    await context.globalState.update(SETUP_COMPLETED_KEY, true);
    await context.globalState.update(SETUP_DISMISSED_KEY, true);
    await restartServer();
    vscode.window.showInformationMessage('EasyCite: 设置向导已完成。');
}

async function configureDependencyInstallMode(): Promise<void> {
    const choice = await vscode.window.showQuickPick<WizardChoice>(
        [
            {
                label: '缺依赖时询问',
                description: '推荐。安装到扩展私有目录，不污染 base/系统 Python',
                action: 'ask',
            },
            {
                label: '缺依赖时自动安装',
                description: '以后自动安装到扩展私有目录',
                action: 'always',
            },
            {
                label: '从不自动安装',
                description: '只提示缺失依赖，由用户自行处理',
                action: 'never',
            },
        ],
        {
            title: 'EasyCite: Python 依赖安装策略',
            placeHolder: '选择缺少依赖时如何处理',
            ignoreFocusOut: true,
        },
    );
    if (!choice) {
        return;
    }
    await vscode.workspace
        .getConfiguration('citation-agent')
        .update('dependencyInstallMode', choice.action, vscode.ConfigurationTarget.Global);
}

async function showPythonRuntimeSummary(context: vscode.ExtensionContext, namespace: string): Promise<void> {
    const status = await checkPythonRuntime(context, namespace);
    traceInfo(
        `EasyCite setup Python status: interpreter=${status.interpreter.join(' ')}, privateLibDir=${
            status.privateLibDir
        }, missingRequired=${status.missingRequired.join(', ') || 'none'}`,
    );
    vscode.window.setStatusBarMessage(`EasyCite: Python 已就绪，私有依赖目录 ${shortPath(status.privateLibDir)}`, 5000);
}

async function configureZoteroAndBetterBibTeX(): Promise<void> {
    const choice = await vscode.window.showQuickPick<WizardChoice>(
        [
            {
                label: '启用 Zotero + Better BibTeX',
                description: '推荐。读取 Zotero 候选并自动导出 BibTeX',
                action: 'enable',
            },
            {
                label: '暂不启用',
                description: '只使用当前 TeX 工程已有 .bib',
                action: 'skip',
            },
        ],
        {
            title: 'EasyCite: Zotero 文献库',
            placeHolder: '是否启用 Zotero 候选来源？',
            ignoreFocusOut: true,
        },
    );
    if (!choice || choice.action === 'skip') {
        return;
    }

    const config = vscode.workspace.getConfiguration('citation-agent');
    const endpoint = await vscode.window.showInputBox({
        title: 'EasyCite: Zotero Local API',
        value: config.get<string>('zoteroEndpoint', '') || 'http://127.0.0.1:23119/api',
        prompt: 'Zotero 本地 API 地址。Better BibTeX JSON-RPC 会自动使用同一端口。',
        ignoreFocusOut: true,
    });
    if (!endpoint) {
        return;
    }

    await config.update('zoteroEnabled', true, vscode.ConfigurationTarget.Workspace);
    await config.update('zoteroEndpoint', endpoint, vscode.ConfigurationTarget.Workspace);

    let zoteroReady = await isZoteroAvailable(endpoint);
    if (!zoteroReady) {
        zoteroReady = await ensureZoteroAvailable();
    }
    if (!zoteroReady) {
        return;
    }

    const bbtReady = await isBetterBibTeXAvailable(endpoint);
    if (bbtReady) {
        vscode.window.showInformationMessage('EasyCite: Zotero 和 Better BibTeX 均可访问。');
    } else {
        vscode.window.showWarningMessage(
            'EasyCite: Zotero 可访问，但 Better BibTeX JSON-RPC 不可用。请确认 Zotero 已安装 Better BibTeX 插件，然后重新运行设置向导。',
        );
    }
}

async function configureModelApi(
    context: vscode.ExtensionContext,
    namespace: string,
    restartServer: () => Promise<void>,
): Promise<void> {
    const config = vscode.workspace.getConfiguration('citation-agent');
    const hasApiKey = Boolean(await context.secrets.get(API_KEY_SECRET));
    const useLLM = config.get<boolean>('useLLM', false);
    const choice = await vscode.window.showQuickPick<WizardChoice>(
        [
            {
                label: hasApiKey && useLLM ? '保留当前模型配置' : '配置模型 API',
                description: hasApiKey && useLLM ? '已检测到 API key' : '启用 LLM 证据判断、查询规划和中文对照',
                action: hasApiKey && useLLM ? 'keep' : 'configure',
            },
            {
                label: '暂不用外部模型',
                description: '使用本地规则；推荐质量会下降',
                action: 'disable',
            },
        ],
        {
            title: 'EasyCite: 模型 API',
            placeHolder: '是否配置外部模型服务？',
            ignoreFocusOut: true,
        },
    );
    if (!choice || choice.action === 'keep') {
        return;
    }
    if (choice.action === 'disable') {
        await config.update('useLLM', false, vscode.ConfigurationTarget.Global);
        return;
    }

    const installed = await ensureOptionalDependency(
        context,
        namespace,
        'llm',
        'EasyCite: LLM 功能需要 Python 包 litellm，是否安装到扩展私有目录？',
    );
    if (!installed) {
        return;
    }
    await configureModelProvider(context, restartServer);
}

async function configureEmbedding(context: vscode.ExtensionContext, namespace: string): Promise<void> {
    const choice = await vscode.window.showQuickPick<WizardChoice>(
        [
            {
                label: '启用本地 embedding 检索',
                description: '推荐。CPU ONNX 语义检索，用于 evidence passage 排序',
                action: 'fastembed',
            },
            {
                label: '启用开发用 hash embedding',
                description: '不下载模型，仅用于离线调试，不具备真实语义能力',
                action: 'hash',
            },
            {
                label: '暂不启用',
                action: 'skip',
            },
        ],
        {
            title: 'EasyCite: Embedding 模型',
            placeHolder: '选择 evidence 语义检索方式',
            ignoreFocusOut: true,
        },
    );
    if (!choice || choice.action === 'skip') {
        return;
    }

    const config = vscode.workspace.getConfiguration('citation-agent');
    await config.update('embeddingEnabled', true, vscode.ConfigurationTarget.Workspace);
    await config.update('embeddingModelProvider', choice.action, vscode.ConfigurationTarget.Workspace);
    if (choice.action === 'hash') {
        return;
    }

    const installed = await ensureOptionalDependency(
        context,
        namespace,
        'embedding',
        'EasyCite: embedding 功能需要 Python 包 fastembed，是否安装到扩展私有目录？',
    );
    if (!installed) {
        await config.update('embeddingEnabled', false, vscode.ConfigurationTarget.Workspace);
        return;
    }

    const modelId = await vscode.window.showInputBox({
        title: 'EasyCite: Embedding 模型 ID',
        value: config.get<string>('embeddingModelId', 'BAAI/bge-small-en-v1.5'),
        prompt: '默认模型适合英文论文检索。首次使用会下载到本机缓存。',
        ignoreFocusOut: true,
    });
    if (modelId) {
        await config.update('embeddingModelId', modelId, vscode.ConfigurationTarget.Workspace);
    }

    const defaultCacheDir = path.join(vscode.env.appRoot ? path.dirname(vscode.env.appRoot) : '', '');
    const cacheDir = await vscode.window.showInputBox({
        title: 'EasyCite: Embedding 缓存目录',
        value:
            config.get<string>('embeddingCacheDir', '') ||
            path.join(process.env.HOME || process.env.USERPROFILE || '', '.cache', 'citation-agent', 'embeddings'),
        prompt: '模型文件会下载到这里。留空则使用默认 ~/.cache/citation-agent/embeddings。',
        ignoreFocusOut: true,
    });
    if (cacheDir !== undefined) {
        await config.update('embeddingCacheDir', cacheDir || '', vscode.ConfigurationTarget.Workspace);
    }
    void defaultCacheDir;

    const autoDownload = await vscode.window.showQuickPick<WizardChoice>(
        [
            {
                label: '首次使用时自动下载模型',
                description: '推荐。由 fastembed 下载模型到缓存目录',
                action: 'always',
            },
            {
                label: '下载前询问/只用本地缓存',
                description: '没有缓存时回退到词法检索',
                action: 'ask',
            },
            {
                label: '永不自动下载',
                action: 'never',
            },
        ],
        {
            title: 'EasyCite: Embedding 模型下载策略',
            placeHolder: '选择模型文件下载方式',
            ignoreFocusOut: true,
        },
    );
    if (autoDownload) {
        await config.update('embeddingAutoDownload', autoDownload.action, vscode.ConfigurationTarget.Workspace);
    }
}

async function configurePdfReader(context: vscode.ExtensionContext, namespace: string): Promise<void> {
    const choice = await vscode.window.showQuickPick<WizardChoice>(
        [
            {
                label: '安装 PDF 阅读增强依赖',
                description: '推荐。安装 PyMuPDF，用于读取 Zotero 附件 PDF 的 evidence passage',
                action: 'install',
            },
            {
                label: '暂不安装',
                description: '仍可使用标题、摘要和 BibTeX 元数据',
                action: 'skip',
            },
        ],
        {
            title: 'EasyCite: PDF 证据读取',
            placeHolder: '是否安装 PDF 读取依赖？',
            ignoreFocusOut: true,
        },
    );
    if (!choice || choice.action === 'skip') {
        return;
    }
    await ensureOptionalDependency(
        context,
        namespace,
        'pdf',
        'EasyCite: PDF 阅读需要 Python 包 PyMuPDF，是否安装到扩展私有目录？',
    );
}

async function ensureOptionalDependency(
    context: vscode.ExtensionContext,
    namespace: string,
    group: 'llm' | 'pdf' | 'embedding',
    prompt: string,
): Promise<boolean> {
    const status = await checkPythonRuntime(context, namespace);
    if (status.missingOptional[group].length === 0) {
        return true;
    }
    const mode = vscode.workspace.getConfiguration('citation-agent').get<string>('dependencyInstallMode', 'ask');
    let shouldInstall = mode === 'always';
    if (mode === 'never') {
        vscode.window.showWarningMessage(`EasyCite: 缺少 ${status.missingOptional[group].join(', ')}。`);
        return false;
    }
    if (!shouldInstall) {
        const choice = await vscode.window.showWarningMessage(prompt, { modal: true }, '安装', '跳过');
        shouldInstall = choice === '安装';
    }
    if (!shouldInstall) {
        return false;
    }
    return installOptionalPackageGroup(context, namespace, group);
}

function shortPath(value: string): string {
    const home = process.env.HOME || process.env.USERPROFILE;
    if (home && value.startsWith(home)) {
        return `~${value.slice(home.length)}`;
    }
    return value;
}
