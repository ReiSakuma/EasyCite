// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as fs from 'fs';
import * as path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { ConfigurationChangeEvent, ConfigurationScope, WorkspaceConfiguration, WorkspaceFolder } from 'vscode';
import { EXTENSION_ROOT_DIR } from './constants';
import { getInterpreterDetails } from './python';
import { getConfiguration, getWorkspaceFolders } from './vscodeapi';

const execFileAsync = promisify(execFile);

export interface ISettings {
    cwd: string;
    workspace: string;
    args: string[];
    path: string[];
    interpreter: string[];
    importStrategy: string;
    showNotifications: string;
}

export function getExtensionSettings(namespace: string, includeInterpreter?: boolean): Promise<ISettings[]> {
    return Promise.all(getWorkspaceFolders().map((w) => getWorkspaceSettings(namespace, w, includeInterpreter)));
}

function resolveVariables(value: string[], workspace?: WorkspaceFolder): string[] {
    const substitutions = new Map<string, string>();
    const home = process.env.HOME || process.env.USERPROFILE;
    if (home) {
        substitutions.set('${userHome}', home);
    }
    if (workspace) {
        substitutions.set('${workspaceFolder}', workspace.uri.fsPath);
    }
    substitutions.set('${cwd}', process.cwd());
    getWorkspaceFolders().forEach((w) => {
        substitutions.set('${workspaceFolder:' + w.name + '}', w.uri.fsPath);
    });

    return value.map((s) => {
        for (const [key, value] of substitutions) {
            s = s.replace(key, value);
        }
        return s;
    });
}

export function getInterpreterFromSetting(namespace: string, scope?: ConfigurationScope) {
    const config = getConfiguration(namespace, scope);
    return config.get<string[]>('interpreter');
}

function getCwd(config: WorkspaceConfiguration, workspace: WorkspaceFolder): string {
    const cwd = config.get<string>('cwd', '${workspaceFolder}');
    return resolveVariables([cwd], workspace)[0];
}

function getProjectVenvInterpreter(): string[] {
    const candidates =
        process.platform === 'win32'
            ? [path.join(EXTENSION_ROOT_DIR, '.venv', 'Scripts', 'python.exe')]
            : [path.join(EXTENSION_ROOT_DIR, '.venv', 'bin', 'python')];

    const interpreter = candidates.find((candidate) => fs.existsSync(candidate));
    return interpreter ? [interpreter] : [];
}

function getCandidatePythonInterpreters(workspace?: WorkspaceFolder): string[] {
    const candidates: string[] = [];
    const push = (candidate: string | undefined) => {
        if (candidate && !candidates.includes(candidate)) {
            candidates.push(candidate);
        }
    };

    if (workspace) {
        push(
            process.platform === 'win32'
                ? path.join(workspace.uri.fsPath, '.venv', 'Scripts', 'python.exe')
                : path.join(workspace.uri.fsPath, '.venv', 'bin', 'python'),
        );
    }

    for (const root of [process.env.CONDA_PREFIX, process.env.VIRTUAL_ENV]) {
        if (!root) {
            continue;
        }
        push(process.platform === 'win32' ? path.join(root, 'python.exe') : path.join(root, 'bin', 'python'));
    }

    const home = process.env.HOME || process.env.USERPROFILE;
    if (home) {
        if (process.platform === 'win32') {
            push(path.join(home, 'miniconda3', 'python.exe'));
            push(path.join(home, 'anaconda3', 'python.exe'));
        } else {
            push(path.join(home, 'miniconda3', 'bin', 'python'));
            push(path.join(home, 'miniconda3', 'bin', 'python3'));
            push(path.join(home, 'anaconda3', 'bin', 'python'));
            push(path.join(home, 'anaconda3', 'bin', 'python3'));
        }
    }

    push(process.platform === 'win32' ? 'python.exe' : 'python3.13');
    push(process.platform === 'win32' ? 'python.exe' : 'python3.12');
    push(process.platform === 'win32' ? 'python.exe' : 'python3.11');
    push(process.platform === 'win32' ? 'python.exe' : 'python3.10');
    push(process.platform === 'win32' ? 'py.exe' : 'python3');
    return candidates;
}

async function isSupportedPythonInterpreter(command: string): Promise<boolean> {
    try {
        const result = await execFileAsync(command, ['--version'], { timeout: 3000 });
        const versionText = `${result.stdout} ${result.stderr}`;
        const match = versionText.match(/Python\s+(\d+)\.(\d+)/);
        if (!match) {
            return false;
        }
        const major = Number(match[1]);
        const minor = Number(match[2]);
        const bundledRequirement = getBundledPythonMinorRequirement();
        if (bundledRequirement) {
            return major === bundledRequirement.major && minor === bundledRequirement.minor;
        }
        return major === 3 && minor >= 10;
    } catch {
        return false;
    }
}

function getBundledPythonMinorRequirement(): { major: number; minor: number } | undefined {
    const pydanticCoreDir = path.join(EXTENSION_ROOT_DIR, 'bundled', 'libs', 'pydantic_core');
    if (!fs.existsSync(pydanticCoreDir)) {
        return undefined;
    }
    for (const fileName of fs.readdirSync(pydanticCoreDir)) {
        const match = fileName.match(/\.cpython-(\d)(\d{2})[-_.]/);
        if (match) {
            return { major: Number(match[1]), minor: Number(match[2]) };
        }
    }
    return undefined;
}

async function findSupportedPythonInterpreter(workspace?: WorkspaceFolder): Promise<string[]> {
    for (const candidate of getCandidatePythonInterpreters(workspace)) {
        if (await isSupportedPythonInterpreter(candidate)) {
            return [candidate];
        }
    }
    return [];
}

async function getInterpreterWithFallback(namespace: string, workspace?: WorkspaceFolder): Promise<string[]> {
    const configured = getInterpreterFromSetting(namespace, workspace) ?? [];
    if (configured.length > 0) {
        return configured;
    }

    const workspaceOrKnownPython = await findSupportedPythonInterpreter(workspace);
    if (workspaceOrKnownPython.length > 0) {
        return workspaceOrKnownPython;
    }

    const projectVenv = getProjectVenvInterpreter();
    if (projectVenv.length > 0) {
        return projectVenv;
    }

    const pythonExtensionInterpreter = (await getInterpreterDetails(workspace?.uri)).path ?? [];
    if (pythonExtensionInterpreter.length > 0 && (await isSupportedPythonInterpreter(pythonExtensionInterpreter[0]))) {
        return pythonExtensionInterpreter;
    }
    return [];
}

export async function getWorkspaceSettings(
    namespace: string,
    workspace: WorkspaceFolder,
    includeInterpreter?: boolean,
): Promise<ISettings> {
    const config = getConfiguration(namespace, workspace.uri);

    let interpreter: string[] = [];
    if (includeInterpreter) {
        interpreter = await getInterpreterWithFallback(namespace, workspace);
    }

    const workspaceSetting = {
        cwd: getCwd(config, workspace),
        workspace: workspace.uri.toString(),
        args: resolveVariables(config.get<string[]>(`args`) ?? [], workspace),
        path: resolveVariables(config.get<string[]>(`path`) ?? [], workspace),
        interpreter: resolveVariables(interpreter, workspace),
        importStrategy: config.get<string>(`importStrategy`) ?? 'useBundled',
        showNotifications: config.get<string>(`showNotifications`) ?? 'off',
    };
    return workspaceSetting;
}

function getGlobalValue<T>(config: WorkspaceConfiguration, key: string, defaultValue: T): T {
    const inspect = config.inspect<T>(key);
    return inspect?.globalValue ?? inspect?.defaultValue ?? defaultValue;
}

export async function getGlobalSettings(namespace: string, includeInterpreter?: boolean): Promise<ISettings> {
    const config = getConfiguration(namespace);

    let interpreter: string[] = [];
    if (includeInterpreter) {
        interpreter = getGlobalValue<string[]>(config, 'interpreter', []);
        if (interpreter === undefined || interpreter.length === 0) {
            interpreter = getProjectVenvInterpreter();
        }
        if (interpreter.length === 0) {
            interpreter = (await getInterpreterDetails()).path ?? [];
        }
    }

    const setting = {
        cwd: process.cwd(),
        workspace: process.cwd(),
        args: getGlobalValue<string[]>(config, 'args', []),
        path: getGlobalValue<string[]>(config, 'path', []),
        interpreter: interpreter,
        importStrategy: getGlobalValue<string>(config, 'importStrategy', 'useBundled'),
        showNotifications: getGlobalValue<string>(config, 'showNotifications', 'off'),
    };
    return setting;
}

export function checkIfConfigurationChanged(e: ConfigurationChangeEvent, namespace: string): boolean {
    const settings = [
        `${namespace}.args`,
        `${namespace}.cwd`,
        `${namespace}.path`,
        `${namespace}.interpreter`,
        `${namespace}.importStrategy`,
        `${namespace}.showNotifications`,
        `${namespace}.dependencyInstallMode`,
        `${namespace}.useLLM`,
        `${namespace}.provider`,
        `${namespace}.model`,
        `${namespace}.apiBase`,
        `${namespace}.timeoutSeconds`,
        `${namespace}.evidenceTimeoutSeconds`,
        `${namespace}.evidenceMaxLLMCandidates`,
        `${namespace}.agentRuntime`,
        `${namespace}.maxSuggestions`,
        `${namespace}.maxMetadataCandidates`,
        `${namespace}.maxAbstractCandidates`,
        `${namespace}.maxEvidencePassages`,
        `${namespace}.enableEvidenceVerification`,
        `${namespace}.ragEnabled`,
        `${namespace}.ragIndexPath`,
        `${namespace}.ragMaxDocuments`,
        `${namespace}.ragChildChunkTokens`,
        `${namespace}.ragChildChunkMaxTokens`,
        `${namespace}.ragChunkOverlapTokens`,
        `${namespace}.ragParentChunkTokens`,
        `${namespace}.ragSemanticChunkingEnabled`,
        `${namespace}.ragSemanticBoundaryThreshold`,
        `${namespace}.ragSemanticHardBoundaryThreshold`,
        `${namespace}.ragSemanticMinParentTokens`,
        `${namespace}.ragSemanticAdaptiveEnabled`,
        `${namespace}.ragSemanticAdaptivePercentile`,
        `${namespace}.ragSemanticAdaptiveMaxThreshold`,
        `${namespace}.ragSemanticMinScoreDrop`,
        `${namespace}.ragSemanticContextWindow`,
        `${namespace}.ragSemanticEmbeddingMinTokens`,
        `${namespace}.ragEmbeddingBatchSize`,
        `${namespace}.ragPaperFusedTopK`,
        `${namespace}.ragPassageFusedTopK`,
        `${namespace}.ragPdfIndexTopK`,
        `${namespace}.ragRerankerEnabled`,
        `${namespace}.ragRerankerModelId`,
        `${namespace}.ragRerankerInputK`,
        `${namespace}.ragRerankerTopK`,
        `${namespace}.embeddingEnabled`,
        `${namespace}.embeddingModelProvider`,
        `${namespace}.embeddingModelId`,
        `${namespace}.embeddingCacheDir`,
        `${namespace}.embeddingAutoDownload`,
        `${namespace}.embeddingTopK`,
        `${namespace}.hybridLexicalWeight`,
        `${namespace}.hybridEmbeddingWeight`,
        `${namespace}.zoteroEnabled`,
        `${namespace}.zoteroEndpoint`,
        `${namespace}.zoteroAutoOpenWithoutPrompt`,
        `${namespace}.zoteroLaunchCommand`,
        `${namespace}.zoteroCollectionKeys`,
        `${namespace}.debugTargetPapers`,
        `${namespace}.memoryEnabled`,
        `${namespace}.memoryPath`,
        `${namespace}.memoryPositiveBoost`,
        `${namespace}.memoryNegativePenalty`,
        `${namespace}.memoryMinSimilarity`,
    ];
    const changed = settings.map((s) => e.affectsConfiguration(s));
    return changed.includes(true);
}
