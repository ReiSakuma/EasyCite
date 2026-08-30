import * as vscode from 'vscode';
import { traceError, traceInfo } from '../common/log/logging';
import { ensureZoteroAvailable, listZoteroCollections, ZoteroCollection } from '../common/zotero';

interface CollectionPickItem extends vscode.QuickPickItem {
    scope: 'all' | 'collection';
    key?: string;
}

const DEFAULT_ZOTERO_ENDPOINT = 'http://127.0.0.1:23119/api';

export function registerConfigureZoteroScopeCommand(
    context: vscode.ExtensionContext,
    restartServer: () => Promise<void>,
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('citation-agent.configureZoteroScope', async () => {
            await configureZoteroScope(restartServer);
        }),
    );
}

async function configureZoteroScope(restartServer: () => Promise<void>): Promise<void> {
    const config = vscode.workspace.getConfiguration('citation-agent');
    let endpoint = config.get<string>('zoteroEndpoint', '').trim();
    if (!endpoint) {
        endpoint = DEFAULT_ZOTERO_ENDPOINT;
        await config.update('zoteroEndpoint', endpoint, vscode.ConfigurationTarget.Workspace);
    }
    if (!config.get<boolean>('zoteroEnabled', false)) {
        await config.update('zoteroEnabled', true, vscode.ConfigurationTarget.Workspace);
    }

    if (!(await ensureZoteroAvailable())) {
        return;
    }

    let collections: ZoteroCollection[] = [];
    try {
        collections = await listZoteroCollections(endpoint);
    } catch (error) {
        traceError(`EasyCite: failed to list Zotero collections: ${error}`);
    }
    if (collections.length === 0) {
        vscode.window.showWarningMessage('EasyCite: 没有从 Zotero 读取到 collection，将继续使用全库范围。');
        await config.update('zoteroCollectionKeys', [], vscode.ConfigurationTarget.Workspace);
        await restartServer();
        return;
    }

    const currentKeys = new Set(config.get<string[]>('zoteroCollectionKeys', []));
    const choices: CollectionPickItem[] = [
        {
            label: '全部 Zotero 文库',
            description: '不限定 collection',
            scope: 'all',
            picked: currentKeys.size === 0,
        },
        ...collections.map((collection) => ({
            label: collectionPath(collection, collections),
            description: collection.key,
            scope: 'collection' as const,
            key: collection.key,
            picked: currentKeys.has(collection.key),
        })),
    ];

    const selected = await vscode.window.showQuickPick(choices, {
        title: 'EasyCite: 配置 Zotero 文献范围',
        placeHolder: '选择一个或多个 collection；选择“全部 Zotero 文库”表示不限定范围',
        canPickMany: true,
        ignoreFocusOut: true,
        matchOnDescription: true,
    });
    if (!selected) {
        return;
    }

    const useWholeLibrary = selected.some((item) => item.scope === 'all');
    const selectedKeys = useWholeLibrary
        ? []
        : selected.filter((item) => item.scope === 'collection' && item.key).map((item) => item.key as string);

    await config.update('zoteroCollectionKeys', selectedKeys, vscode.ConfigurationTarget.Workspace);
    traceInfo(
        selectedKeys.length > 0
            ? `EasyCite: Zotero scope set to collection keys: ${selectedKeys.join(', ')}`
            : 'EasyCite: Zotero scope reset to whole library.',
    );
    vscode.window.showInformationMessage(
        selectedKeys.length > 0
            ? `EasyCite: 已限定到 ${selectedKeys.length} 个 Zotero collection。`
            : 'EasyCite: 已切换为全部 Zotero 文库。',
    );
    await restartServer();
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
