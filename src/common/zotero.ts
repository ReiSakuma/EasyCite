import * as childProcess from 'child_process';
import * as http from 'http';
import * as https from 'https';
import * as vscode from 'vscode';
import { traceError, traceInfo } from './log/logging';

const ZOTERO_CHECK_TIMEOUT_MS = 1200;
const ZOTERO_LAUNCH_WAIT_MS = 10000;
const ZOTERO_POLL_INTERVAL_MS = 600;

interface ZoteroPromptItem extends vscode.QuickPickItem {
    action: 'open' | 'openAndRemember' | 'cancel';
}

export interface ZoteroCollection {
    key: string;
    name: string;
    parentKey?: string;
}

export async function ensureZoteroAvailable(): Promise<boolean> {
    const config = vscode.workspace.getConfiguration('citation-agent');
    const zoteroEnabled = config.get<boolean>('zoteroEnabled', false);
    const endpoint = config.get<string>('zoteroEndpoint', '');
    if (!zoteroEnabled || !endpoint) {
        return true;
    }

    if (await isZoteroAvailable(endpoint)) {
        return true;
    }

    const autoOpen = config.get<boolean>('zoteroAutoOpenWithoutPrompt', false);
    if (!autoOpen) {
        const choice = await vscode.window.showQuickPick<ZoteroPromptItem>(
            [
                {
                    label: '授权并打开 Zotero',
                    description: '本次启动 Zotero 并读取本地文献库',
                    action: 'open',
                },
                {
                    label: '授权并打开 Zotero，不再提示',
                    description: '以后检测到 Zotero 未打开时自动启动',
                    action: 'openAndRemember',
                },
                {
                    label: '取消',
                    description: '不启动 Zotero，本次不继续 Zotero 候选检索',
                    action: 'cancel',
                },
            ],
            {
                title: 'EasyCite: 尚未打开 Zotero，是否授权获取 Zotero 的信息？',
                placeHolder: '选择是否启动 Zotero 本地服务',
                ignoreFocusOut: true,
            },
        );
        if (!choice || choice.action === 'cancel') {
            vscode.window.showInformationMessage('EasyCite: 已取消 Zotero 候选检索。');
            return false;
        }
        if (choice.action === 'openAndRemember') {
            await config.update('zoteroAutoOpenWithoutPrompt', true, vscode.ConfigurationTarget.Workspace);
        }
    }

    traceInfo('EasyCite: Zotero Local API unavailable; trying to launch Zotero.');
    await launchZotero(config);
    const available = await waitForZotero(endpoint);
    if (!available) {
        vscode.window.showErrorMessage(
            'EasyCite: 已尝试打开 Zotero，但本地 API 仍不可用。请检查 Zotero 是否安装并启用了本地 API。',
        );
        return false;
    }
    vscode.window.setStatusBarMessage('EasyCite: Zotero 已启动。', 5000);
    return true;
}

export async function isZoteroAvailable(endpoint: string): Promise<boolean> {
    const url = buildZoteroProbeUrl(endpoint);
    return new Promise((resolve) => {
        const client = url.protocol === 'https:' ? https : http;
        const request = client.request(
            url,
            {
                method: 'GET',
                timeout: ZOTERO_CHECK_TIMEOUT_MS,
                headers: {
                    ['Zotero-API-Version']: '3',
                },
            },
            (response) => {
                response.resume();
                resolve(Boolean(response.statusCode && response.statusCode >= 200 && response.statusCode < 500));
            },
        );
        request.on('timeout', () => {
            request.destroy();
            resolve(false);
        });
        request.on('error', () => resolve(false));
        request.end();
    });
}

export async function isBetterBibTeXAvailable(endpoint: string): Promise<boolean> {
    const url = buildBetterBibTeXProbeUrl(endpoint);
    return new Promise((resolve) => {
        const client = url.protocol === 'https:' ? https : http;
        const payload = JSON.stringify({
            jsonrpc: '2.0',
            method: 'item.citationkey',
            params: [[]],
            id: 'citation-agent-setup',
        });
        const headers: Record<string, string | number> = {};
        headers.Accept = 'application/json';
        headers['Content-Type'] = 'application/json';
        headers['Content-Length'] = Buffer.byteLength(payload);
        const request = client.request(
            url,
            {
                method: 'POST',
                timeout: ZOTERO_CHECK_TIMEOUT_MS,
                headers,
            },
            (response) => {
                response.resume();
                resolve(Boolean(response.statusCode && response.statusCode >= 200 && response.statusCode < 500));
            },
        );
        request.on('timeout', () => {
            request.destroy();
            resolve(false);
        });
        request.on('error', () => resolve(false));
        request.write(payload);
        request.end();
    });
}

export async function listZoteroCollections(endpoint: string): Promise<ZoteroCollection[]> {
    const collections: ZoteroCollection[] = [];
    const base = normalizeZoteroApiBase(endpoint);
    const firstUrl = new URL(`${base}/users/0/collections?limit=100&format=json`);
    let start = 0;
    for (;;) {
        firstUrl.searchParams.set('start', String(start));
        const payload = await requestJson(firstUrl, ZOTERO_LAUNCH_WAIT_MS);
        if (!Array.isArray(payload)) {
            break;
        }
        for (const item of payload) {
            if (!item || typeof item !== 'object') {
                continue;
            }
            const data = typeof item.data === 'object' && item.data ? item.data : item;
            const key = String(item.key || data.key || '').trim();
            const name = String(data.name || '').trim();
            if (!key || !name) {
                continue;
            }
            const parent = String(data.parentCollection || '').trim();
            collections.push({
                key,
                name,
                parentKey: parent || undefined,
            });
        }
        if (payload.length < 100) {
            break;
        }
        start += payload.length;
    }
    return collections.sort((a, b) => collectionPath(a, collections).localeCompare(collectionPath(b, collections)));
}

async function waitForZotero(endpoint: string): Promise<boolean> {
    const deadline = Date.now() + ZOTERO_LAUNCH_WAIT_MS;
    while (Date.now() < deadline) {
        if (await isZoteroAvailable(endpoint)) {
            return true;
        }
        await delay(ZOTERO_POLL_INTERVAL_MS);
    }
    return false;
}

async function launchZotero(config: vscode.WorkspaceConfiguration): Promise<void> {
    const customCommand = config.get<string>('zoteroLaunchCommand', '').trim();
    if (customCommand) {
        spawnDetached(customCommand, [], true);
        return;
    }

    const commands = defaultLaunchCommands();
    for (const command of commands) {
        try {
            spawnDetached(command.command, command.args);
            return;
        } catch (error) {
            traceError(`EasyCite: failed to launch Zotero with ${command.command}: ${error}`);
        }
    }
}

function defaultLaunchCommands(): Array<{ command: string; args: string[] }> {
    if (process.platform === 'win32') {
        return [
            { command: 'powershell.exe', args: windowsBackgroundLaunchArgs() },
            { command: 'cmd.exe', args: ['/c', 'start', '/min', '', 'zotero'] },
            { command: 'cmd.exe', args: ['/c', 'start', '', 'zotero://select/library'] },
        ];
    }
    if (process.platform === 'darwin') {
        return [
            { command: 'open', args: ['-g', '-j', '-a', 'Zotero'] },
            { command: 'open', args: ['zotero://select/library'] },
        ];
    }
    if (isWsl()) {
        return [
            { command: 'powershell.exe', args: windowsBackgroundLaunchArgs() },
            { command: 'cmd.exe', args: ['/c', 'start', '/min', '', 'zotero'] },
            { command: 'cmd.exe', args: ['/c', 'start', '', 'zotero://select/library'] },
        ];
    }
    return [
        { command: 'zotero', args: [] },
        { command: 'xdg-open', args: ['zotero://select/library'] },
    ];
}

function windowsBackgroundLaunchArgs(): string[] {
    const command = [
        '$paths = @(',
        '"$env:LOCALAPPDATA\\Zotero\\zotero.exe",',
        '"$env:ProgramFiles\\Zotero\\zotero.exe",',
        '"${env:ProgramFiles(x86)}\\Zotero\\zotero.exe"',
        ');',
        '$exe = $paths | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1;',
        'if (-not $exe) { $cmd = Get-Command zotero.exe -ErrorAction SilentlyContinue; if ($cmd) { $exe = $cmd.Source } }',
        'if ($exe) { Start-Process -FilePath $exe -WindowStyle Minimized }',
        'else { Start-Process -FilePath "zotero://select/library" -WindowStyle Minimized }',
    ].join(' ');
    return ['-NoProfile', '-WindowStyle', 'Hidden', '-Command', command];
}

function spawnDetached(command: string, args: string[], shell = false): void {
    const child = childProcess.spawn(command, args, {
        detached: true,
        stdio: 'ignore',
        shell,
    });
    child.unref();
}

function buildZoteroProbeUrl(endpoint: string): URL {
    const base = normalizeZoteroApiBase(endpoint);
    return new URL(`${base}/users/0/items/top?limit=1&format=json`);
}

function buildBetterBibTeXProbeUrl(endpoint: string): URL {
    const parsed = new URL(endpoint);
    parsed.pathname = '/better-bibtex/json-rpc';
    parsed.search = '';
    parsed.hash = '';
    return parsed;
}

function normalizeZoteroApiBase(endpoint: string): string {
    const base = endpoint.replace(/\/+$/, '');
    return base.endsWith('/api') ? base : `${base}/api`;
}

async function requestJson(url: URL, timeoutMs: number): Promise<unknown | undefined> {
    return new Promise((resolve) => {
        const client = url.protocol === 'https:' ? https : http;
        const request = client.request(
            url,
            {
                method: 'GET',
                timeout: timeoutMs,
                headers: {
                    Accept: 'application/json',
                    ['Zotero-API-Version']: '3',
                },
            },
            (response) => {
                const chunks: Buffer[] = [];
                response.on('data', (chunk: Buffer) => chunks.push(chunk));
                response.on('end', () => {
                    if (!response.statusCode || response.statusCode < 200 || response.statusCode >= 300) {
                        resolve(undefined);
                        return;
                    }
                    try {
                        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
                    } catch (error) {
                        traceError(`EasyCite: failed to parse Zotero JSON: ${error}`);
                        resolve(undefined);
                    }
                });
            },
        );
        request.on('timeout', () => {
            request.destroy();
            resolve(undefined);
        });
        request.on('error', () => resolve(undefined));
        request.end();
    });
}

function collectionPath(collection: ZoteroCollection, all: ZoteroCollection[]): string {
    const byKey = new Map(all.map((item) => [item.key, item]));
    const names = [collection.name];
    let parentKey = collection.parentKey;
    const seen = new Set<string>([collection.key]);
    while (parentKey && !seen.has(parentKey)) {
        const parent = byKey.get(parentKey);
        if (!parent) {
            break;
        }
        names.unshift(parent.name);
        seen.add(parent.key);
        parentKey = parent.parentKey;
    }
    return names.join(' / ');
}

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function isWsl(): boolean {
    return Boolean(process.env.WSL_DISTRO_NAME || process.env.WSL_INTEROP);
}
