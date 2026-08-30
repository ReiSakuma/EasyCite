import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

export interface TexProjectContext {
    workspaceRoot: string;
    currentFile: string;
    rootFile: string;
    rootReason: string;
    includedFiles: string[];
    bibliographyOwner: vscode.Uri;
    bibliographyText: string;
    bibliographyReferences: string[];
}

export async function analyzeTexProject(document: vscode.TextDocument): Promise<TexProjectContext> {
    const currentFile = document.uri.fsPath;
    const workspaceRoot = vscode.workspace.getWorkspaceFolder(document.uri)?.uri.fsPath ?? path.dirname(currentFile);
    const currentText = document.getText();
    const rootFile = await findRootTexFile(document, workspaceRoot, currentText);
    const rootText = rootFile === currentFile ? currentText : await readText(rootFile);
    const includedFiles = rootFile ? await collectIncludedTexFiles(rootFile, rootText, workspaceRoot) : [];
    const includedTexts = await Promise.all(includedFiles.map(async (file) => ({ file, text: await readText(file) })));
    const bibliographyText = [rootText || currentText, ...includedTexts.map((item) => item.text)].join('\n');
    const bibliographyOwner = vscode.Uri.file(rootFile || currentFile);
    const bibliographyReferences = uniquePaths([
        ...resolveBibReferences(rootText || currentText, path.dirname(bibliographyOwner.fsPath)),
        ...includedTexts.flatMap((item) => resolveBibReferences(item.text, path.dirname(item.file))),
    ]);

    return {
        workspaceRoot,
        currentFile,
        rootFile: rootFile || currentFile,
        rootReason: rootFile === currentFile ? '当前 TeX 文件是主文件' : '由 TeX 工程解析识别到的主文件',
        includedFiles,
        bibliographyOwner,
        bibliographyText,
        bibliographyReferences,
    };
}

export function resolveBibReferences(texContent: string, baseDir: string): string[] {
    const paths: string[] = [];
    const addBibResource = /\\addbibresource(?:\[[^\]]*\])?\{([^}]+)\}/g;
    const bibliography = /\\bibliography\{([^}]+)\}/g;
    let match = addBibResource.exec(texContent);
    while (match) {
        paths.push(resolveBibPath(match[1], baseDir));
        match = addBibResource.exec(texContent);
    }
    match = bibliography.exec(texContent);
    while (match) {
        for (const item of match[1].split(',')) {
            paths.push(resolveBibPath(item.trim(), baseDir));
        }
        match = bibliography.exec(texContent);
    }
    return uniquePaths(paths);
}

export function resolveBibPath(rawPath: string, baseDir: string): string {
    const normalized = rawPath.trim();
    const withExtension = path.extname(normalized) ? normalized : `${normalized}.bib`;
    return path.isAbsolute(withExtension) ? withExtension : path.resolve(baseDir, withExtension);
}

function isMainTex(text: string): boolean {
    return /\\documentclass(?:\[[^\]]*\])?\{[^}]+\}/.test(text) || /\\begin\{document\}/.test(text);
}

async function findRootTexFile(
    document: vscode.TextDocument,
    workspaceRoot: string,
    currentText: string,
): Promise<string> {
    const currentFile = document.uri.fsPath;
    const rootDirective = resolveRootDirective(currentText, path.dirname(currentFile));
    if (rootDirective && rootDirective.startsWith(workspaceRoot) && fs.existsSync(rootDirective)) {
        return rootDirective;
    }
    if (isMainTex(currentText)) {
        return currentFile;
    }

    const texFiles = await vscode.workspace.findFiles(
        new vscode.RelativePattern(workspaceRoot, '**/*.tex'),
        '**/{.git,.venv,node_modules,dist,out}/**',
        120,
    );
    const candidates: Array<{ file: string; score: number }> = [];
    for (const uri of texFiles) {
        if (uri.fsPath === currentFile) {
            continue;
        }
        const text = await readText(uri.fsPath);
        if (!isMainTex(text)) {
            continue;
        }
        const includesCurrent = parseIncludedTexPaths(text, path.dirname(uri.fsPath)).includes(currentFile);
        candidates.push({
            file: uri.fsPath,
            score: rootCandidateScore(uri.fsPath, currentFile, includesCurrent),
        });
    }

    candidates.sort((first, second) => first.score - second.score);
    return candidates[0]?.file ?? currentFile;
}

async function collectIncludedTexFiles(rootFile: string, rootText: string, workspaceRoot: string): Promise<string[]> {
    const seen = new Set<string>();
    const queue = parseIncludedTexPaths(rootText, path.dirname(rootFile));

    while (queue.length > 0 && seen.size < 200) {
        const file = queue.shift();
        if (!file || seen.has(file) || !file.startsWith(workspaceRoot) || !fs.existsSync(file)) {
            continue;
        }
        seen.add(file);
        const text = await readText(file);
        queue.push(...parseIncludedTexPaths(text, path.dirname(file)));
    }

    return [...seen];
}

function parseIncludedTexPaths(texContent: string, baseDir: string): string[] {
    const paths: string[] = [];
    const includePattern = /\\(?:input|include|subfile)\s*\{([^}]+)\}/g;
    let match = includePattern.exec(texContent);
    while (match) {
        paths.push(resolveTexPath(match[1], baseDir));
        match = includePattern.exec(texContent);
    }
    const importPattern = /\\(?:import|subimport|inputfrom|subinputfrom)\s*\{([^}]+)\}\s*\{([^}]+)\}/g;
    match = importPattern.exec(texContent);
    while (match) {
        paths.push(resolveTexPath(match[2], path.resolve(baseDir, match[1].trim())));
        match = importPattern.exec(texContent);
    }
    return uniquePaths(paths);
}

function resolveTexPath(rawPath: string, baseDir: string): string {
    const normalized = rawPath.trim();
    const withExtension = path.extname(normalized) ? normalized : `${normalized}.tex`;
    return path.isAbsolute(withExtension) ? withExtension : path.resolve(baseDir, withExtension);
}

function rootCandidateScore(rootFile: string, currentFile: string, includesCurrent: boolean): number {
    const basename = path.basename(rootFile).toLowerCase();
    const sameTree = currentFile.startsWith(path.dirname(rootFile)) ? 0 : 10;
    const nameScore = ['main.tex', 'paper.tex', 'manuscript.tex', 'article.tex'].includes(basename) ? 0 : 3;
    return (includesCurrent ? -50 : 0) + sameTree + nameScore;
}

function resolveRootDirective(texContent: string, baseDir: string): string | undefined {
    const lines = texContent.split(/\r?\n/).slice(0, 20);
    for (const line of lines) {
        const match = line.match(/%\s*!\s*T[eE]X\s+root\s*=\s*(.+)$/);
        if (!match) {
            continue;
        }
        return resolveTexPath(match[1].trim(), baseDir);
    }
    return undefined;
}

async function readText(filePath: string): Promise<string> {
    try {
        return await fs.promises.readFile(filePath, 'utf-8');
    } catch {
        return '';
    }
}

function uniquePaths(paths: string[]): string[] {
    const seen = new Set<string>();
    const unique: string[] = [];
    for (const item of paths) {
        const normalized = path.normalize(item);
        if (!normalized || seen.has(normalized)) {
            continue;
        }
        seen.add(normalized);
        unique.push(normalized);
    }
    return unique;
}
