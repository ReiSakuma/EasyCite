import * as path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs-extra';
import * as vscode from 'vscode';
import { BUNDLED_PYTHON_SCRIPTS_DIR } from './constants';
import { traceError, traceInfo } from './log/logging';
import { getWorkspaceSettings } from './settings';
import { getProjectRoot } from './utilities';

const execFileAsync = promisify(execFile);

export const PRIVATE_LIBS_ENV = 'CITATION_AGENT_PRIVATE_LIBS';
const REQUIRED_MODULE_CHECK_TIMEOUT_MS = 60 * 1000;
const OPTIONAL_MODULE_CHECK_TIMEOUT_MS = 45 * 1000;

interface PythonPackage {
    module: string;
    packageName: string;
}

const REQUIRED_PACKAGES: PythonPackage[] = [
    { module: 'pydantic', packageName: 'pydantic>=2,<3' },
    { module: 'pydantic_core', packageName: 'pydantic-core>=2,<3' },
    { module: 'pygls', packageName: 'pygls>=2,<3' },
    { module: 'lsprotocol', packageName: 'lsprotocol>=2025,<2026' },
    { module: 'langgraph', packageName: 'langgraph>=0.2,<1' },
];

export const OPTIONAL_PACKAGE_GROUPS = {
    llm: [{ module: 'litellm', packageName: 'litellm' }],
    pdf: [{ module: 'fitz', packageName: 'pymupdf' }],
    embedding: [{ module: 'fastembed', packageName: 'fastembed' }],
} satisfies Record<string, PythonPackage[]>;

export interface PythonRuntimeCheck {
    interpreter: string[];
    privateLibDir: string;
    missingRequired: string[];
    missingOptional: Record<keyof typeof OPTIONAL_PACKAGE_GROUPS, string[]>;
    errors: Record<string, string>;
}

export function initializePrivatePythonLibEnvironment(context: vscode.ExtensionContext): string {
    const privateLibDir = path.join(context.globalStorageUri.fsPath, 'python-libs');
    fs.ensureDirSync(privateLibDir);
    process.env[PRIVATE_LIBS_ENV] = privateLibDir;
    return privateLibDir;
}

export async function checkPythonRuntime(
    context: vscode.ExtensionContext,
    namespace: string,
): Promise<PythonRuntimeCheck> {
    const privateLibDir = initializePrivatePythonLibEnvironment(context);
    const workspace = await getProjectRoot();
    const settings = await getWorkspaceSettings(namespace, workspace, true);
    const interpreter = settings.interpreter;
    if (interpreter.length === 0) {
        return {
            interpreter,
            privateLibDir,
            missingRequired: REQUIRED_PACKAGES.map((item) => item.module),
            missingOptional: { llm: [], pdf: [], embedding: [] },
            errors: { python: 'No compatible Python interpreter was found.' },
        };
    }

    const requiredModules = REQUIRED_PACKAGES.map((item) => item.module);
    const requiredResult = await checkPythonModules(
        interpreter,
        privateLibDir,
        requiredModules,
        REQUIRED_MODULE_CHECK_TIMEOUT_MS,
    );
    const optionalModules = [
        ...OPTIONAL_PACKAGE_GROUPS.llm.map((item) => item.module),
        ...OPTIONAL_PACKAGE_GROUPS.pdf.map((item) => item.module),
        ...OPTIONAL_PACKAGE_GROUPS.embedding.map((item) => item.module),
    ];
    const optionalResult = await checkPythonModules(
        interpreter,
        privateLibDir,
        optionalModules,
        OPTIONAL_MODULE_CHECK_TIMEOUT_MS,
    );
    return {
        interpreter,
        privateLibDir,
        missingRequired: REQUIRED_PACKAGES.map((item) => item.module).filter((module) =>
            requiredResult.missing.includes(module),
        ),
        missingOptional: {
            llm: OPTIONAL_PACKAGE_GROUPS.llm
                .map((item) => item.module)
                .filter((module) => optionalResult.missing.includes(module)),
            pdf: OPTIONAL_PACKAGE_GROUPS.pdf
                .map((item) => item.module)
                .filter((module) => optionalResult.missing.includes(module)),
            embedding: OPTIONAL_PACKAGE_GROUPS.embedding
                .map((item) => item.module)
                .filter((module) => optionalResult.missing.includes(module)),
        },
        errors: { ...requiredResult.errors, ...optionalResult.errors },
    };
}

export async function ensureRequiredPythonRuntime(
    context: vscode.ExtensionContext,
    namespace: string,
): Promise<boolean> {
    const status = await checkPythonRuntime(context, namespace);
    if (status.interpreter.length === 0) {
        vscode.window.showErrorMessage(
            'EasyCite: 没有找到兼容的 Python。通用/Linux 版请安装 Python 3.13；Windows 版推荐 Python 3.11 至 3.13。',
        );
        return false;
    }
    if (status.missingRequired.length === 0) {
        traceInfo(`EasyCite Python runtime ready: ${status.interpreter.join(' ')}`);
        return true;
    }

    const mode = vscode.workspace.getConfiguration('citation-agent').get<string>('dependencyInstallMode', 'ask');
    if (mode === 'never') {
        vscode.window.showErrorMessage(
            `EasyCite: Python 必需依赖缺失：${status.missingRequired.join(', ')}。请运行设置向导修复。`,
        );
        return false;
    }

    let shouldInstall = mode === 'always';
    if (!shouldInstall) {
        const choice = await vscode.window.showWarningMessage(
            `EasyCite: Python 必需依赖缺失：${status.missingRequired.join(', ')}。是否安装到扩展私有目录？`,
            { modal: true },
            '安装',
            '稍后',
        );
        shouldInstall = choice === '安装';
    }
    if (!shouldInstall) {
        return false;
    }

    const packages = REQUIRED_PACKAGES.filter((item) => status.missingRequired.includes(item.module)).map(
        (item) => item.packageName,
    );
    const installed = await installPythonPackages(status.interpreter, status.privateLibDir, packages);
    if (!installed) {
        return false;
    }
    const recheck = await checkPythonRuntime(context, namespace);
    if (recheck.missingRequired.length > 0) {
        vscode.window.showErrorMessage(
            `EasyCite: 依赖安装后仍缺失：${recheck.missingRequired.join(', ')}。请查看输出日志。`,
        );
        return false;
    }
    return true;
}

export async function installOptionalPackageGroup(
    context: vscode.ExtensionContext,
    namespace: string,
    group: keyof typeof OPTIONAL_PACKAGE_GROUPS,
): Promise<boolean> {
    const status = await checkPythonRuntime(context, namespace);
    const missing = status.missingOptional[group];
    if (missing.length === 0) {
        return true;
    }
    const packages = OPTIONAL_PACKAGE_GROUPS[group]
        .filter((item) => missing.includes(item.module))
        .map((item) => item.packageName);
    return installPythonPackages(status.interpreter, status.privateLibDir, packages);
}

export async function ensureConfiguredOptionalPythonRuntime(
    context: vscode.ExtensionContext,
    namespace: string,
): Promise<void> {
    const config = vscode.workspace.getConfiguration(namespace);
    const desiredGroups: (keyof typeof OPTIONAL_PACKAGE_GROUPS)[] = [];
    if (config.get<boolean>('useLLM', false)) {
        desiredGroups.push('llm');
    }
    if (config.get<boolean>('enableEvidenceVerification', true)) {
        desiredGroups.push('pdf');
    }
    if (config.get<boolean>('embeddingEnabled', false)) {
        desiredGroups.push('embedding');
    }
    if (desiredGroups.length === 0) {
        return;
    }

    const status = await checkPythonRuntime(context, namespace);
    const missingPackages: string[] = [];
    for (const group of desiredGroups) {
        const missingModules = status.missingOptional[group];
        const packages = OPTIONAL_PACKAGE_GROUPS[group]
            .filter((item) => missingModules.includes(item.module))
            .map((item) => item.packageName);
        missingPackages.push(...packages);
    }

    const uniquePackages = [...new Set(missingPackages)];
    if (uniquePackages.length === 0) {
        return;
    }

    const mode = config.get<string>('dependencyInstallMode', 'ask');
    if (mode === 'never') {
        vscode.window.showWarningMessage(
            `EasyCite: 当前功能需要可选 Python 依赖：${uniquePackages.join(
                ', ',
            )}。dependencyInstallMode=never，因此不会自动安装。`,
        );
        return;
    }

    let shouldInstall = mode === 'always';
    if (!shouldInstall) {
        const choice = await vscode.window.showWarningMessage(
            `EasyCite: 当前已启用的功能需要 Python 依赖：${uniquePackages.join(', ')}。是否安装到扩展私有目录？`,
            { modal: true },
            '安装',
            '稍后',
        );
        shouldInstall = choice === '安装';
    }
    if (!shouldInstall) {
        return;
    }

    const installed = await installPythonPackages(status.interpreter, status.privateLibDir, uniquePackages);
    if (!installed) {
        return;
    }
    const recheck = await checkPythonRuntime(context, namespace);
    const stillMissing = desiredGroups.flatMap((group) => recheck.missingOptional[group]);
    if (stillMissing.length > 0) {
        vscode.window.showWarningMessage(
            `EasyCite: 可选依赖安装后仍缺失：${[...new Set(stillMissing)].join(
                ', ',
            )}。相关功能会降级，请查看输出日志。`,
        );
    }
}

async function installPythonPackages(interpreter: string[], targetDir: string, packages: string[]): Promise<boolean> {
    if (interpreter.length === 0 || packages.length === 0) {
        return false;
    }
    await fs.ensureDir(targetDir);
    try {
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: `EasyCite: 正在安装 Python 依赖 ${packages.join(', ')}`,
                cancellable: false,
            },
            async () => {
                await execFileAsync(
                    interpreter[0],
                    ['-m', 'pip', 'install', '--upgrade', '--target', targetDir, ...packages],
                    {
                        timeout: 10 * 60 * 1000,
                        maxBuffer: 1024 * 1024 * 20,
                        env: process.env,
                    },
                );
            },
        );
        traceInfo(`EasyCite installed Python packages into ${targetDir}: ${packages.join(', ')}`);
        return true;
    } catch (error) {
        traceError(`EasyCite failed to install Python packages: ${error}`);
        vscode.window.showErrorMessage(`EasyCite: Python 依赖安装失败。请查看输出日志，或检查网络/pip 是否可用。`);
        return false;
    }
}

async function checkPythonModules(
    interpreter: string[],
    privateLibDir: string,
    modules: string[],
    timeoutMs: number,
): Promise<{ missing: string[]; errors: Record<string, string> }> {
    if (modules.length === 0) {
        return { missing: [], errors: {} };
    }
    const bundledLibDir = path.join(BUNDLED_PYTHON_SCRIPTS_DIR, 'libs');
    const script = [
        'import importlib, json, os, sys',
        'paths = json.loads(os.environ.get("CITATION_AGENT_CHECK_PATHS", "[]"))',
        'if paths:',
        '    bundled = paths[0]',
        '    if bundled and bundled not in sys.path: sys.path.insert(0, bundled)',
        '    for p in paths[1:]:',
        '        if p and p not in sys.path: sys.path.append(p)',
        'modules = json.loads(os.environ["CITATION_AGENT_CHECK_MODULES"])',
        'missing = []',
        'errors = {}',
        'for module in modules:',
        '    try:',
        '        importlib.import_module(module)',
        '    except Exception as exc:',
        '        missing.append(module)',
        '        errors[module] = f"{type(exc).__name__}: {exc}"',
        'print(json.dumps({"missing": missing, "errors": errors}))',
    ].join('\n');
    try {
        const env = { ...process.env };
        env.CITATION_AGENT_CHECK_MODULES = JSON.stringify(modules);
        env.CITATION_AGENT_CHECK_PATHS = JSON.stringify([bundledLibDir, privateLibDir]);
        const result = await execFileAsync(interpreter[0], ['-c', script], {
            timeout: timeoutMs,
            maxBuffer: 1024 * 1024 * 4,
            env,
        });
        const parsed = parseModuleCheckOutput(result.stdout, result.stderr);
        if (parsed) {
            return parsed;
        }
        throw new Error(`Module check did not return JSON. stdout=${result.stdout} stderr=${result.stderr}`);
    } catch (error: any) {
        const stdout = typeof error?.stdout === 'string' && error.stdout.trim() ? ` stdout=${error.stdout}` : '';
        const stderr = typeof error?.stderr === 'string' && error.stderr.trim() ? ` stderr=${error.stderr}` : '';
        traceError(
            `EasyCite Python module check failed for [${modules.join(', ')}] after ${timeoutMs}ms: ${error}${stdout}${stderr}`,
        );
        return { missing: modules, errors: { python: String(error) } };
    }
}

function parseModuleCheckOutput(
    stdout: string,
    stderr: string,
): { missing: string[]; errors: Record<string, string> } | undefined {
    const combined = [stdout, stderr].filter(Boolean).join('\n');
    const candidates = combined
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.startsWith('{') && line.endsWith('}'));
    for (const candidate of candidates.reverse()) {
        try {
            const parsed = JSON.parse(candidate) as { missing?: unknown; errors?: unknown };
            if (Array.isArray(parsed.missing) && parsed.errors && typeof parsed.errors === 'object') {
                return {
                    missing: parsed.missing.map((item) => String(item)),
                    errors: parsed.errors as Record<string, string>,
                };
            }
        } catch {
            continue;
        }
    }
    return undefined;
}
