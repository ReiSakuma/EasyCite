import * as vscode from 'vscode';
import { traceInfo } from '../common/log/logging';

export const API_KEY_SECRET = 'citation-agent.apiKey';

interface ProviderPreset {
    label: string;
    provider: string;
    model: string;
    apiBase?: string;
}

const PRESETS: ProviderPreset[] = [
    { label: 'OpenAI', provider: 'openai', model: 'gpt-4o-mini' },
    { label: 'DeepSeek', provider: 'deepseek', model: 'deepseek/deepseek-chat' },
    { label: 'OpenRouter', provider: 'openrouter', model: 'openrouter/openai/gpt-4o-mini' },
    { label: 'OpenAI-Compatible', provider: 'custom', model: 'openai/gpt-4o-mini' },
];

export async function applyModelEnvironment(context: vscode.ExtensionContext): Promise<void> {
    const config = vscode.workspace.getConfiguration('citation-agent');
    const useLLM = config.get<boolean>('useLLM', false);
    const provider = config.get<string>('provider', 'openai');
    const model = config.get<string>('model', 'gpt-4o-mini');
    const apiBase = config.get<string>('apiBase', '');
    const timeoutSeconds = config.get<number>('timeoutSeconds', 60);
    const evidenceTimeoutSeconds = config.get<number>('evidenceTimeoutSeconds', 60);
    const evidenceMaxLLMCandidates = config.get<number>('evidenceMaxLLMCandidates', 8);
    const contextQueryBudgetTokens = config.get<number>('contextQueryBudgetTokens', 3200);
    const contextEvidenceBudgetTokens = config.get<number>('contextEvidenceBudgetTokens', 9000);
    const contextTranslationBudgetTokens = config.get<number>('contextTranslationBudgetTokens', 5000);
    const agentRuntime = config.get<string>('agentRuntime', 'langgraph');
    const maxSuggestions = config.get<number>('maxSuggestions', 5);
    const maxMetadataCandidates = config.get<number>('maxMetadataCandidates', 50);
    const maxAbstractCandidates = config.get<number>('maxAbstractCandidates', 8);
    const maxEvidencePassages = config.get<number>('maxEvidencePassages', 5);
    const enableEvidenceVerification = config.get<boolean>('enableEvidenceVerification', true);
    const zoteroEnabled = config.get<boolean>('zoteroEnabled', false);
    const zoteroEndpoint = config.get<string>('zoteroEndpoint', '');
    const zoteroCollectionKeys = config.get<string[]>('zoteroCollectionKeys', []);
    const embeddingEnabled = config.get<boolean>('embeddingEnabled', false);
    const embeddingModelProvider = config.get<string>('embeddingModelProvider', 'fastembed');
    const embeddingModelId = config.get<string>('embeddingModelId', 'BAAI/bge-small-en-v1.5');
    const embeddingCacheDir = config.get<string>('embeddingCacheDir', '');
    const embeddingAutoDownload = config.get<string>('embeddingAutoDownload', 'ask');
    const embeddingTopK = config.get<number>('embeddingTopK', 12);
    const hybridLexicalWeight = config.get<number>('hybridLexicalWeight', 0.35);
    const hybridEmbeddingWeight = config.get<number>('hybridEmbeddingWeight', 0.65);
    const debugTargetPapers = config.get<string[]>('debugTargetPapers', []);
    const memoryEnabled = config.get<boolean>('memoryEnabled', true);
    const memoryPath = config.get<string>('memoryPath', '');
    const memoryPositiveBoost = config.get<number>('memoryPositiveBoost', 0.08);
    const memoryNegativePenalty = config.get<number>('memoryNegativePenalty', 0.18);
    const memoryMinSimilarity = config.get<number>('memoryMinSimilarity', 0.22);

    process.env.CITATION_AGENT_USE_LLM = useLLM ? 'true' : 'false';
    process.env.CITATION_AGENT_PROVIDER = provider;
    process.env.CITATION_AGENT_MODEL = model;
    process.env.CITATION_AGENT_API_BASE = apiBase;
    process.env.CITATION_AGENT_TIMEOUT_SECONDS = String(timeoutSeconds);
    process.env.CITATION_AGENT_EVIDENCE_TIMEOUT_SECONDS = String(evidenceTimeoutSeconds);
    process.env.CITATION_AGENT_EVIDENCE_MAX_LLM_CANDIDATES = String(evidenceMaxLLMCandidates);
    process.env.CITATION_AGENT_CONTEXT_QUERY_BUDGET_TOKENS = String(contextQueryBudgetTokens);
    process.env.CITATION_AGENT_CONTEXT_EVIDENCE_BUDGET_TOKENS = String(contextEvidenceBudgetTokens);
    process.env.CITATION_AGENT_CONTEXT_TRANSLATION_BUDGET_TOKENS = String(contextTranslationBudgetTokens);
    process.env.CITATION_AGENT_AGENT_RUNTIME = agentRuntime;
    process.env.CITATION_AGENT_MAX_SUGGESTIONS = String(maxSuggestions);
    process.env.CITATION_AGENT_MAX_METADATA_CANDIDATES = String(maxMetadataCandidates);
    process.env.CITATION_AGENT_MAX_ABSTRACT_CANDIDATES = String(maxAbstractCandidates);
    process.env.CITATION_AGENT_MAX_EVIDENCE_PASSAGES = String(maxEvidencePassages);
    process.env.CITATION_AGENT_ENABLE_EVIDENCE_VERIFICATION = enableEvidenceVerification ? 'true' : 'false';
    process.env.CITATION_AGENT_ZOTERO_ENABLED = zoteroEnabled ? 'true' : 'false';
    process.env.CITATION_AGENT_ZOTERO_ENDPOINT = zoteroEndpoint;
    process.env.CITATION_AGENT_ZOTERO_COLLECTION_KEYS = JSON.stringify(
        zoteroCollectionKeys.filter((item) => item.trim()),
    );
    process.env.CITATION_AGENT_EMBEDDING_ENABLED = embeddingEnabled ? 'true' : 'false';
    process.env.CITATION_AGENT_EMBEDDING_MODEL_PROVIDER = embeddingModelProvider;
    process.env.CITATION_AGENT_EMBEDDING_MODEL_ID = embeddingModelId;
    process.env.CITATION_AGENT_EMBEDDING_CACHE_DIR = embeddingCacheDir;
    process.env.CITATION_AGENT_EMBEDDING_AUTO_DOWNLOAD = embeddingAutoDownload;
    process.env.CITATION_AGENT_EMBEDDING_TOP_K = String(embeddingTopK);
    process.env.CITATION_AGENT_HYBRID_LEXICAL_WEIGHT = String(hybridLexicalWeight);
    process.env.CITATION_AGENT_HYBRID_EMBEDDING_WEIGHT = String(hybridEmbeddingWeight);
    process.env.CITATION_AGENT_DEBUG_TARGET_PAPERS = JSON.stringify(debugTargetPapers.filter((item) => item.trim()));
    process.env.CITATION_AGENT_MEMORY_ENABLED = memoryEnabled ? 'true' : 'false';
    process.env.CITATION_AGENT_MEMORY_PATH = memoryPath;
    process.env.CITATION_AGENT_MEMORY_POSITIVE_BOOST = String(memoryPositiveBoost);
    process.env.CITATION_AGENT_MEMORY_NEGATIVE_PENALTY = String(memoryNegativePenalty);
    process.env.CITATION_AGENT_MEMORY_MIN_SIMILARITY = String(memoryMinSimilarity);

    const apiKey = await context.secrets.get(API_KEY_SECRET);
    if (apiKey) {
        process.env.CITATION_AGENT_API_KEY = apiKey;
    } else {
        delete process.env.CITATION_AGENT_API_KEY;
    }
}

export function registerConfigureProviderCommand(
    context: vscode.ExtensionContext,
    restartServer: () => Promise<void>,
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('citation-agent.configureProvider', async () => {
            await configureModelProvider(context, restartServer);
        }),
    );
}

export async function configureModelProvider(
    context: vscode.ExtensionContext,
    restartServer: () => Promise<void>,
): Promise<boolean> {
    const preset = await vscode.window.showQuickPick(PRESETS, {
        title: 'EasyCite: 配置模型服务',
        placeHolder: '选择模型 Provider',
    });
    if (!preset) {
        return false;
    }

    const model = await vscode.window.showInputBox({
        title: 'EasyCite: 模型名称',
        value: preset.model,
        prompt: '使用 LiteLLM 模型命名，例如 gpt-4o-mini、deepseek/deepseek-chat、openrouter/...',
    });
    if (!model) {
        return false;
    }

    const apiBase = await vscode.window.showInputBox({
        title: 'EasyCite: API Base URL',
        value: preset.apiBase ?? '',
        prompt: 'OpenAI-Compatible 或私有网关可填写；官方 Provider 可留空。',
    });

    const apiKey = await vscode.window.showInputBox({
        title: 'EasyCite: API Key',
        password: true,
        ignoreFocusOut: true,
        prompt: 'API Key 会保存到 VS Code SecretStorage，不写入 settings.json 或项目文件。',
    });
    if (!apiKey) {
        return false;
    }

    const config = vscode.workspace.getConfiguration('citation-agent');
    await config.update('useLLM', true, vscode.ConfigurationTarget.Global);
    await config.update('provider', preset.provider, vscode.ConfigurationTarget.Global);
    await config.update('model', model, vscode.ConfigurationTarget.Global);
    await config.update('apiBase', apiBase ?? '', vscode.ConfigurationTarget.Global);
    await context.secrets.store(API_KEY_SECRET, apiKey);

    await applyModelEnvironment(context);
    traceInfo(`EasyCite model configured: provider=${preset.provider}, model=${model}`);
    await restartServer();
    vscode.window.showInformationMessage('EasyCite: 模型服务已配置并重启后端。');
    return true;
}
