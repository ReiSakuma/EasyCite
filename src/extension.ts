// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as vscode from 'vscode';
import { State } from 'vscode-languageclient';
import { LanguageClient } from 'vscode-languageclient/node';
import { registerAuditSelectionCommand } from './commands/auditSelection';
import { applyModelEnvironment, registerConfigureProviderCommand } from './commands/configureProvider';
import { registerConfigureZoteroScopeCommand } from './commands/configureZoteroScope';
import { maybeShowFirstRunSetup, registerSetupWizardCommand } from './commands/setupWizard';
import { registerToolHealthCommand } from './commands/toolHealth';
import { registerLogger, traceError, traceLog, traceVerbose } from './common/log/logging';
import {
    checkVersion,
    getInterpreterDetails,
    initializePython,
    onDidChangePythonInterpreter,
    resolveInterpreter,
} from './common/python';
import { restartServer } from './common/server';
import {
    ensureConfiguredOptionalPythonRuntime,
    ensureRequiredPythonRuntime,
    initializePrivatePythonLibEnvironment,
} from './common/runtimeEnvironment';
import { checkIfConfigurationChanged, getInterpreterFromSetting } from './common/settings';
import { loadServerDefaults } from './common/setup';
import { LS_SERVER_RESTART_DELAY } from './common/constants';
import { getLSClientTraceLevel } from './common/utilities';
import { createOutputChannel, onDidChangeConfiguration, registerCommand } from './common/vscodeapi';

let lsClient: LanguageClient | undefined;
let isRestarting = false;
let restartTimer: NodeJS.Timeout | undefined;
export async function activate(context: vscode.ExtensionContext): Promise<void> {
    // This is required to get server name and module. This should be
    // the first thing that we do in this extension.
    const serverInfo = loadServerDefaults();
    const serverName = serverInfo.name;
    const serverId = serverInfo.module;

    // Setup logging
    const outputChannel = createOutputChannel(serverName);
    context.subscriptions.push(outputChannel, registerLogger(outputChannel));
    initializePrivatePythonLibEnvironment(context);

    const changeLogLevel = async (c: vscode.LogLevel, g: vscode.LogLevel) => {
        const level = getLSClientTraceLevel(c, g);
        await lsClient?.setTrace(level);
    };

    context.subscriptions.push(
        outputChannel.onDidChangeLogLevel(async (e) => {
            await changeLogLevel(e, vscode.env.logLevel);
        }),
        vscode.env.onDidChangeLogLevel(async (e) => {
            await changeLogLevel(outputChannel.logLevel, e);
        }),
    );

    // Log Server information
    traceLog(`Name: ${serverInfo.name}`);
    traceLog(`Module: ${serverInfo.module}`);
    traceVerbose(`Full Server Info: ${JSON.stringify(serverInfo)}`);

    const runServer = async () => {
        if (isRestarting) {
            if (restartTimer) {
                clearTimeout(restartTimer);
            }
            restartTimer = setTimeout(runServer, LS_SERVER_RESTART_DELAY);
            return;
        }
        isRestarting = true;
        try {
            if (!(await ensureRequiredPythonRuntime(context, serverId))) {
                return;
            }
            await applyModelEnvironment(context);
            await ensureConfiguredOptionalPythonRuntime(context, serverId);
            const interpreter = getInterpreterFromSetting(serverId);
            if (interpreter && interpreter.length > 0) {
                const resolved = await resolveInterpreter(interpreter);
                if (!resolved || checkVersion(resolved)) {
                    traceVerbose(`Using interpreter from ${serverInfo.module}.interpreter: ${interpreter.join(' ')}`);
                    lsClient = await restartServer(serverId, serverName, outputChannel, lsClient);
                }
                return;
            }

            const interpreterDetails = await getInterpreterDetails();
            if (interpreterDetails.path) {
                traceVerbose(`Using interpreter from Python extension: ${interpreterDetails.path.join(' ')}`);
                lsClient = await restartServer(serverId, serverName, outputChannel, lsClient);
                return;
            }

            traceVerbose(`Python extension interpreter unavailable; using EasyCite interpreter fallback.`);
            lsClient = await restartServer(serverId, serverName, outputChannel, lsClient);
        } finally {
            isRestarting = false;
        }
    };

    context.subscriptions.push(
        onDidChangePythonInterpreter(async () => {
            await runServer();
        }),
        onDidChangeConfiguration(async (e: vscode.ConfigurationChangeEvent) => {
            if (checkIfConfigurationChanged(e, serverId)) {
                if (restartTimer) {
                    clearTimeout(restartTimer);
                }
                restartTimer = setTimeout(() => {
                    restartTimer = undefined;
                    void runServer();
                }, LS_SERVER_RESTART_DELAY);
            }
        }),
        registerCommand(`${serverId}.restart`, async () => {
            await runServer();
        }),
    );

    registerAuditSelectionCommand(
        context,
        () => lsClient,
        async () => {
            if (restartTimer) {
                clearTimeout(restartTimer);
                restartTimer = undefined;
                await runServer();
            }
            if (!lsClient || lsClient.state !== State.Running) {
                await runServer();
            }
            return lsClient;
        },
    );

    registerConfigureProviderCommand(context, async () => {
        await runServer();
    });
    registerConfigureZoteroScopeCommand(context, async () => {
        await runServer();
    });
    registerSetupWizardCommand(context, serverId, async () => {
        await runServer();
    });
    registerToolHealthCommand(
        context,
        () => lsClient,
        async () => {
            if (restartTimer) {
                clearTimeout(restartTimer);
                restartTimer = undefined;
                await runServer();
            }
            if (!lsClient || lsClient.state !== State.Running) {
                await runServer();
            }
            return lsClient;
        },
    );

    setImmediate(async () => {
        await applyModelEnvironment(context);
        if (!(await ensureRequiredPythonRuntime(context, serverId))) {
            await maybeShowFirstRunSetup(context, serverId, async () => {
                await runServer();
            });
            return;
        }
        const interpreter = getInterpreterFromSetting(serverId);
        if (interpreter === undefined || interpreter.length === 0) {
            traceLog(`Python extension loading`);
            await initializePython(context.subscriptions);
            traceLog(`Python extension loaded`);
        }
        await runServer();
        await maybeShowFirstRunSetup(context, serverId, async () => {
            await runServer();
        });
    });
}

export async function deactivate(): Promise<void> {
    if (lsClient) {
        try {
            await lsClient.stop();
        } catch (ex) {
            traceError(`Server: Stop failed: ${ex}`);
        }
    }
}
