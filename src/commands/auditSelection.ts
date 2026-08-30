import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { State } from 'vscode-languageclient';
import { LanguageClient } from 'vscode-languageclient/node';
import { traceError, traceInfo, traceLog } from '../common/log/logging';
import { ensureZoteroAvailable } from '../common/zotero';
import { analyzeTexProject, resolveBibReferences, TexProjectContext } from '../common/texProject';
import {
    AuditClaimParams,
    AuditClaimRequest,
    AuditClaimResult,
    CitationSuggestion,
    CitationScanSentence,
    ClearMemoryRequest,
    RankCitationScanRequest,
    RecordFeedbackRequest,
    ResolveBibtexRequest,
    ResolvedBibtexEntry,
    TranslateEvidenceRequest,
    TranslateEvidenceResult,
    TranslateEvidenceText,
} from '../types/protocol';

const AUDIT_REQUEST_TIMEOUT_MS = 600_000;
const TRANSLATION_REQUEST_TIMEOUT_MS = 120_000;
const RESOLVE_BIBTEX_REQUEST_TIMEOUT_MS = 120_000;

type ClientProvider = () => LanguageClient | undefined;

interface CandidateQuickPickItem extends vscode.QuickPickItem {
    itemType: 'candidate' | 'insertSelected' | 'clearSelected' | 'endReview';
    suggestion?: CitationSuggestion;
}

interface CandidateReviewSession {
    documentUri: vscode.Uri;
    selection: vscode.Selection;
    claim: string;
    result: AuditClaimResult;
    client: LanguageClient;
    selectedKeys: Set<string>;
    expandedEvidenceIndex?: number;
}

interface CandidateReviewControls {
    pauseReview: () => void;
    endReview: () => void;
}

interface CandidatePanelMessage {
    command?: 'showEvidence' | 'toggle' | 'insertOne' | 'insertSelected' | 'clearSelected' | 'endReview';
    index?: number;
}

interface DiagnosticPanelMessage {
    command?: 'openRunRecord' | 'saveEvalCase';
    evalCase?: EvalCaseDraft;
}

interface EvalCaseDraft {
    id?: string;
    gold?: string;
    acceptable?: string;
    forbidden?: string;
    notes?: string;
}

interface CitationPlanPanelMessage {
    command?: 'applyFull' | 'citeOnly' | 'cancel';
}

const CITATION_DIAGNOSTIC_SOURCE = 'EasyCite';
const CITATION_DIAGNOSTIC_CODE = 'citation-needed';
const CITATION_AUTO_DIAGNOSTIC_CODE = 'citation-candidate';
const TEX_PROJECT_SCAN_DEBOUNCE_MS = 1500;
const MAX_AUTO_SCAN_FILES = 40;
const MAX_AUTO_SCAN_DIAGNOSTICS_PER_FILE = 80;
const RANK_CITATION_SCAN_TIMEOUT_MS = 20_000;
const DEFAULT_SCAN_EMBEDDING_THRESHOLD = 0.56;

interface CitationScanCandidate {
    id: string;
    range: vscode.Range;
    text: string;
    lexicalScore: number;
    lexicalReason: string;
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

function getWorkspacePath(document: vscode.TextDocument): string {
    return (
        vscode.workspace.getWorkspaceFolder(document.uri)?.uri.fsPath ??
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ??
        ''
    );
}

function getParagraphContext(document: vscode.TextDocument, selection: vscode.Selection): string {
    let start = selection.start.line;
    let end = selection.end.line;

    while (start > 0 && document.lineAt(start - 1).text.trim().length > 0) {
        start -= 1;
    }

    while (end < document.lineCount - 1 && document.lineAt(end + 1).text.trim().length > 0) {
        end += 1;
    }

    const lines: string[] = [];
    for (let line = start; line <= end; line += 1) {
        lines.push(document.lineAt(line).text);
    }
    return lines.join('\n');
}

function extractExistingCitations(text: string): string[] {
    const citationKeys = new Set<string>();
    const citeCommand = /\\(?:[a-zA-Z]*cite[a-zA-Z]*|autocite|parencite|textcite)\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}/g;
    let match = citeCommand.exec(text);

    while (match) {
        for (const key of match[1].split(',')) {
            const normalized = key.trim();
            if (normalized) {
                citationKeys.add(normalized);
            }
        }
        match = citeCommand.exec(text);
    }

    return [...citationKeys];
}

function buildAuditParams(
    document: vscode.TextDocument,
    selection: vscode.Selection,
    forceRefresh = false,
): AuditClaimParams {
    const claim = document.getText(selection).trim();
    return {
        workspace_path: getWorkspacePath(document),
        file_path: document.uri.fsPath,
        line_start: selection.start.line,
        line_end: selection.end.line,
        claim,
        context: getParagraphContext(document, selection),
        existing_citations: extractExistingCitations(claim),
        force_refresh: forceRefresh,
    };
}

function renderSummary(result: AuditClaimResult): string {
    const needed = result.citation_needed ? '需要引用' : '暂不需要新增引用';
    const confidence = `${Math.round(result.confidence * 100)}%`;
    const diagnosticCount = result.diagnostic_suggestions?.length ?? 0;
    const totalCandidates = result.suggestions.length + diagnosticCount;
    const candidateCount =
        totalCandidates > 0 ? `，推荐 ${result.suggestions.length} 条，诊断 ${diagnosticCount} 条` : '';
    return `${needed}，${result.claim_type}，${result.source}，置信度 ${confidence}${candidateCount}：${result.reason}`;
}

function renderSuggestionDescription(suggestion: CitationSuggestion): string {
    const verdictLabels: Record<CitationSuggestion['verdict'], string> = {
        direct_support: '直接支持',
        foundational_support: '奠基支持',
        partial_support: '部分支持',
        related_only: '主题相关',
        metadata_match: '元数据匹配',
        insufficient_evidence: '证据不足',
        contradiction: '可能矛盾',
    };
    const parts = [
        verdictLabels[suggestion.verdict],
        suggestion.year,
        suggestion.source,
        suggestion.citation_key ? `@${suggestion.citation_key}` : '无引用键',
        `${Math.round(suggestion.confidence * 100)}%`,
    ].filter(Boolean);
    return parts.join(' · ');
}

function truncateEvidence(text: string, maxLength = 240): string {
    const normalized = text.replace(/\s+/g, ' ').trim();
    if (normalized.length <= maxLength) {
        return normalized;
    }
    return `${normalized.slice(0, maxLength - 1).trimEnd()}…`;
}

function renderEvidenceLocation(evidence: CitationSuggestion['evidence']): string {
    return [
        evidence.source,
        evidence.page ? `p.${evidence.page}` : undefined,
        evidence.section,
        typeof evidence.score === 'number' ? `score ${evidence.score.toFixed(2)}` : undefined,
        typeof evidence.quality_score === 'number' ? `quality ${evidence.quality_score.toFixed(2)}` : undefined,
    ]
        .filter(Boolean)
        .join(' · ');
}

function renderEvidenceQualityBadges(evidence: CitationSuggestion['evidence']): string {
    const badges: string[] = [];
    if (evidence.is_own_claim === true) {
        badges.push('作者自有论断');
    }
    if (evidence.is_related_work_summary === true) {
        badges.push('Related Work 转述');
    }
    if (evidence.supports_core_mechanism === true) {
        badges.push('覆盖核心机制');
    }
    if (evidence.supports_core_mechanism === false) {
        badges.push('核心机制不足');
    }
    for (const flag of evidence.quality_flags ?? []) {
        if (!['own_claim', 'supports_core_mechanism'].includes(flag)) {
            badges.push(flag);
        }
    }
    return [...new Set(badges)].slice(0, 5).join(' · ');
}

function renderEvidenceSnippet(evidence: CitationSuggestion['evidence'], index?: number): string | undefined {
    if (!evidence.text) {
        return undefined;
    }
    const label = index === undefined ? '最相关文段' : `备选文段 ${index}`;
    const location = renderEvidenceLocation(evidence);
    const quality = renderEvidenceQualityBadges(evidence);
    return `${label}${location ? ` (${location})` : ''}${quality ? ` [${quality}]` : ''}: ${truncateEvidence(evidence.text)}`;
}

function renderEvidencePassages(suggestion: CitationSuggestion): string[] {
    const passages = suggestion.evidence_passages.length > 0 ? suggestion.evidence_passages : [suggestion.evidence];
    const rendered: string[] = [];
    const seen = new Set<string>();

    for (const passage of passages.slice(0, 3)) {
        const text = passage.text?.replace(/\s+/g, ' ').trim();
        if (!text || seen.has(text)) {
            continue;
        }
        seen.add(text);
        const snippet = renderEvidenceSnippet(passage, rendered.length === 0 ? undefined : rendered.length + 1);
        if (snippet) {
            rendered.push(snippet);
        }
    }

    return rendered;
}

function renderInlineEvidence(suggestion: CitationSuggestion): string {
    const passages = suggestion.evidence_passages.length > 0 ? suggestion.evidence_passages : [suggestion.evidence];
    const seen = new Set<string>();
    const evidenceBlocks = passages
        .filter((passage) => passage.text?.trim())
        .filter((passage) => {
            const key = passage.text?.replace(/\s+/g, ' ').trim() ?? '';
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        })
        .map((passage, index) => {
            const location = renderEvidenceLocation(passage);
            const quality = renderEvidenceQualityBadges(passage);
            const label = index === 0 ? '最相关文段' : `备选文段 ${index + 1}`;
            return `
                <section class="full-evidence-item">
                    <div class="full-evidence-head">
                        <strong>${htmlEscape(label)}</strong>
                        ${location ? `<span>${htmlEscape(location)}</span>` : ''}
                    </div>
                    ${quality ? `<p class="quality">${htmlEscape(quality)}</p>` : ''}
                    <p>${htmlEscape(passage.text ?? '')}</p>
                </section>
            `;
        })
        .join('');

    return `
        <div class="full-evidence">
            <div class="full-evidence-summary">
                <strong>判断理由</strong>
                <p>${htmlEscape(suggestion.reason)}</p>
                ${suggestion.citation_key ? `<span class="inline-key">@${htmlEscape(suggestion.citation_key)}</span>` : ''}
            </div>
            ${evidenceBlocks || '<p class="muted">没有可展示的 evidence passage。</p>'}
        </div>
    `;
}

function renderSuggestionDetail(suggestion: CitationSuggestion): string {
    const authors = suggestion.authors.length > 0 ? suggestion.authors.slice(0, 3).join(', ') : '';
    const venue = suggestion.venue ? ` ${suggestion.venue}` : '';
    const header = `${authors}${venue}`.trim();
    const evidencePreview = renderEvidencePassages(suggestion).slice(0, 1);
    const lines = [header, suggestion.reason, ...evidencePreview, '选择后可查看完整证据或插入引用。'].filter(Boolean);
    return lines.join('\n');
}

function renderCandidateWhy(suggestion: CitationSuggestion, diagnostic: boolean): string {
    const trace = suggestion.trace ?? [];
    const traceItems = trace
        .slice(-5)
        .map((line) => `<li>${htmlEscape(line)}</li>`)
        .join('');
    const status = diagnostic ? '这篇文献被系统识别过，但没有进入主推荐区。' : '这篇文献进入了主推荐区。';
    const traceBlock = traceItems
        ? `<ol>${traceItems}</ol>`
        : '<p class="muted">暂无候选生命周期 trace。旧缓存或未经过 v5 trace recorder 的结果可能会出现这种情况。</p>';
    return `
        <details class="why">
            <summary>为什么推荐/未推荐</summary>
            <p>${htmlEscape(status)}</p>
            <p>${htmlEscape(suggestion.reason)}</p>
            ${traceBlock}
        </details>
    `;
}

function htmlEscape(value: string): string {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderCandidatePanelHtml(webview: vscode.Webview, session: CandidateReviewSession): string {
    const nonce = Date.now().toString(36);
    const config = vscode.workspace.getConfiguration('citation-agent');
    const zoteroEnabled = config.get<boolean>('zoteroEnabled', false);
    const zoteroEndpoint = config.get<string>('zoteroEndpoint', '');
    const selectedKeys = [...session.selectedKeys];
    const selectedSummary =
        selectedKeys.length > 0
            ? selectedKeys.map((key) => `<span class="chip">@${htmlEscape(key)}</span>`).join('')
            : '<span class="muted">还没有选择引用</span>';
    const diagnosticSuggestions = session.result.diagnostic_suggestions ?? [];
    const toolWarnings = session.result.tool_warnings ?? [];
    const toolWarningBanner =
        toolWarnings.length > 0
            ? `<div class="tool-warning"><strong>工具降级</strong>${toolWarnings.map((warning) => `<p>${htmlEscape(warning)}</p>`).join('')}</div>`
            : '';
    const renderCards = (suggestions: CitationSuggestion[], startIndex: number, diagnostic: boolean) =>
        suggestions
            .map((suggestion, offset) => {
                const index = startIndex + offset;
                const key = suggestion.citation_key;
                const selected = key ? session.selectedKeys.has(key) : false;
                const expanded = session.expandedEvidenceIndex === index;
                const evidence = renderEvidencePassages(suggestion)
                    .slice(0, 2)
                    .map((line) => `<p class="evidence">${htmlEscape(line)}</p>`)
                    .join('');
                const insertDisabled = key ? '' : 'disabled';
                const toggleDisabled = key ? '' : 'disabled';
                const toggleText = selected ? '移出引用列表' : '加入引用列表';
                const evidenceButtonText = expanded ? '收起证据' : '查看完整证据';
                const keyText = key ? `@${key}` : '无引用键';
                return `
            <article class="card ${selected ? 'selected' : ''} ${diagnostic ? 'diagnostic' : ''}">
                <div class="card-main">
                    <div class="title-row">
                        <h3>${htmlEscape(suggestion.title)}</h3>
                        <div class="badges">
                            ${diagnostic ? '<span class="diagnostic-badge">诊断候选</span>' : ''}
                            ${selected ? '<span class="selected-badge">已选</span>' : ''}
                        </div>
                    </div>
                    <p class="meta">${htmlEscape(renderSuggestionDescription(suggestion))}</p>
                    <p class="authors">${htmlEscape(suggestion.authors.slice(0, 4).join(', '))}</p>
                    <p class="reason">${htmlEscape(suggestion.reason)}</p>
                    ${renderCandidateWhy(suggestion, diagnostic)}
                    ${evidence}
                    <p class="key">${htmlEscape(keyText)}</p>
                </div>
                <div class="actions">
                    <button data-command="showEvidence" data-index="${index}">${evidenceButtonText}</button>
                    <button ${toggleDisabled} data-command="toggle" data-index="${index}">${toggleText}</button>
                    <button ${insertDisabled} data-command="insertOne" data-index="${index}">仅插入此条</button>
                </div>
                ${expanded ? renderInlineEvidence(suggestion) : ''}
            </article>
        `;
            })
            .join('');
    const recommendedCards = renderCards(session.result.suggestions, 0, false);
    const diagnosticCards = renderCards(diagnosticSuggestions, session.result.suggestions.length, true);
    const diagnosticSection =
        diagnosticSuggestions.length > 0
            ? `
            <section class="candidate-section">
                <h2>未入选但已识别的候选</h2>
                <p class="section-note">这些条目被召回或验证过，但没有进入最终推荐前五。可用于排查 MAGIC-SLAM 这类疑似漏选的文献。</p>
                ${diagnosticCards}
            </section>
        `
            : '';
    const recommendedSection = recommendedCards || renderEmptyCandidateState(session, zoteroEnabled, zoteroEndpoint);

    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EasyCite Candidates</title>
    <style>
        body {
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            margin: 0;
            padding: 18px;
        }
        .toolbar {
            position: sticky;
            top: 0;
            z-index: 2;
            background: var(--vscode-editor-background);
            border-bottom: 1px solid var(--vscode-panel-border);
            padding-bottom: 14px;
            margin-bottom: 14px;
        }
        h1 {
            font-size: 18px;
            margin: 0 0 10px;
        }
        .claim {
            border-left: 3px solid var(--vscode-textLink-foreground);
            margin: 0 0 12px;
            padding-left: 10px;
            color: var(--vscode-descriptionForeground);
            line-height: 1.5;
        }
        .selected-row {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }
        .chip,
        .selected-badge,
        .diagnostic-badge {
            background: var(--vscode-badge-background);
            border-radius: 3px;
            color: var(--vscode-badge-foreground);
            padding: 2px 6px;
        }
        .diagnostic-badge {
            background: var(--vscode-editorWarning-foreground);
            color: var(--vscode-editor-background);
        }
        .muted,
        .meta,
        .authors,
        .key {
            color: var(--vscode-descriptionForeground);
        }
        .toolbar-actions,
        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
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
        button:disabled {
            cursor: not-allowed;
            opacity: 0.45;
        }
        .card {
            border: 1px solid var(--vscode-panel-border);
            border-radius: 6px;
            margin-bottom: 12px;
            padding: 12px;
        }
        .card.selected {
            border-color: var(--vscode-textLink-foreground);
        }
        .card.diagnostic {
            border-style: dashed;
        }
        .candidate-section {
            margin-top: 24px;
        }
        h2 {
            border-top: 1px solid var(--vscode-panel-border);
            font-size: 16px;
            margin: 20px 0 8px;
            padding-top: 16px;
        }
        .section-note {
            color: var(--vscode-descriptionForeground);
            margin-bottom: 12px;
        }
        .empty-state {
            border: 1px dashed var(--vscode-panel-border);
            color: var(--vscode-descriptionForeground);
            padding: 12px;
        }
        .target-trace ol {
            background: var(--vscode-editorWidget-background);
            border: 1px solid var(--vscode-panel-border);
            margin: 10px 0 0;
            padding: 10px 10px 10px 28px;
        }
        .target-trace li {
            line-height: 1.5;
            margin: 5px 0;
        }
        .run-trace table {
            border-collapse: collapse;
            margin-top: 10px;
            width: 100%;
        }
        .agent-state-grid {
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            margin: 10px 0;
        }
        .state-box {
            background: var(--vscode-editorWidget-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            padding: 9px 10px;
        }
        .state-box strong {
            display: block;
            margin-bottom: 4px;
        }
        .state-box code {
            overflow-wrap: anywhere;
        }
        .eval-form {
            background: var(--vscode-editorWidget-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            margin-top: 12px;
            padding: 12px;
        }
        .eval-form-grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        }
        .eval-form label {
            color: var(--vscode-descriptionForeground);
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .eval-form input,
        .eval-form textarea {
            background: var(--vscode-input-background);
            border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
            color: var(--vscode-input-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            padding: 6px 8px;
        }
        .eval-form textarea {
            min-height: 58px;
            resize: vertical;
        }
        .eval-form-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .run-trace th,
        .run-trace td {
            border-bottom: 1px solid var(--vscode-panel-border);
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
        }
        .run-trace th {
            color: var(--vscode-descriptionForeground);
            font-weight: 600;
        }
        .run-trace code {
            font-family: var(--vscode-editor-font-family);
        }
        .title-row {
            align-items: start;
            display: flex;
            gap: 10px;
            justify-content: space-between;
        }
        .badges {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            justify-content: flex-end;
        }
        h3 {
            font-size: 15px;
            line-height: 1.35;
            margin: 0;
        }
        p {
            line-height: 1.5;
            margin: 6px 0;
        }
        .reason {
            color: var(--vscode-foreground);
        }
        .why {
            background: var(--vscode-editorWidget-background);
            border-left: 3px solid var(--vscode-textLink-foreground);
            margin: 8px 0;
            padding: 7px 9px;
        }
        .why summary {
            cursor: pointer;
            font-weight: 600;
        }
        .why ol {
            margin: 6px 0 0;
            padding-left: 20px;
        }
        .why li {
            line-height: 1.45;
            margin: 4px 0;
        }
        .evidence {
            background: var(--vscode-textBlockQuote-background);
            border-left: 3px solid var(--vscode-textBlockQuote-border);
            padding: 7px 9px;
        }
        .full-evidence {
            border-top: 1px solid var(--vscode-panel-border);
            margin-top: 12px;
            padding-top: 12px;
        }
        .full-evidence-summary {
            background: var(--vscode-editorWidget-background);
            border-left: 3px solid var(--vscode-textLink-foreground);
            margin-bottom: 10px;
            padding: 8px 10px;
        }
        .full-evidence-summary p {
            margin-bottom: 4px;
        }
        .inline-key {
            color: var(--vscode-descriptionForeground);
            font-family: var(--vscode-editor-font-family);
        }
        .full-evidence-item {
            background: var(--vscode-textBlockQuote-background);
            margin: 8px 0;
            padding: 9px 10px;
        }
        .full-evidence-head {
            align-items: center;
            color: var(--vscode-descriptionForeground);
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 5px;
        }
        .full-evidence-item p {
            margin: 0;
            white-space: pre-wrap;
        }
        .quality {
            color: var(--vscode-descriptionForeground);
            font-size: 12px;
            margin-bottom: 6px !important;
        }
        .actions {
            margin-top: 10px;
        }
        .tool-warning {
            border-left: 3px solid var(--vscode-editorWarning-foreground);
            background: var(--vscode-textBlockQuote-background);
            margin: 12px 0 0;
            padding: 8px 12px;
        }
        .tool-warning p {
            margin: 4px 0 0;
        }
    </style>
</head>
<body>
    <section class="toolbar">
        <h1>EasyCite Candidates</h1>
        <p class="claim">${htmlEscape(session.claim)}</p>
        <div class="selected-row">
            <strong>已选引用</strong>
            ${selectedSummary}
        </div>
        <div class="toolbar-actions">
            <button ${selectedKeys.length > 0 ? '' : 'disabled'} data-command="insertSelected">插入已选择的 ${selectedKeys.length} 条引用</button>
            <button ${selectedKeys.length > 0 ? '' : 'disabled'} data-command="clearSelected">清空已选</button>
            <button data-command="endReview">结束审核</button>
        </div>
        ${toolWarningBanner}
    </section>
    <main>
        <section class="candidate-section">
            <h2>推荐候选</h2>
            ${recommendedSection}
        </section>
        ${diagnosticSection}
    </main>
    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        document.addEventListener('click', (event) => {
            const button = event.target.closest('button[data-command]');
            if (!button || button.disabled) {
                return;
            }
            const indexValue = button.getAttribute('data-index');
            const command = button.getAttribute('data-command');
            vscode.postMessage({
                command,
                index: indexValue === null ? undefined : Number(indexValue),
            });
        });
    </script>
</body>
</html>`;
}

function renderAgentStateSection(
    result: AuditClaimResult,
    evalExportExpanded: boolean,
    developerMode: boolean,
): string {
    const state = result.agent_state;
    if (!state || Object.keys(state).length === 0) {
        return '';
    }
    const counts = state.candidate_pool?.counts ?? {};
    const retrievalPlan = state.retrieval_plan ?? {};
    const model = state.runtime?.model ?? {};
    const graph = state.graph ?? {};
    const contextEngineering = state.context_engineering ?? {};
    const toolReliability = state.tool_reliability ?? {};
    const graphNodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    const graphEdges = Array.isArray(graph.edges) ? graph.edges : [];
    const checkpointPolicy = graph.checkpoint_policy ?? {};
    const checkpointAfter = Array.isArray(checkpointPolicy.checkpoint_after) ? checkpointPolicy.checkpoint_after : [];
    const citationKeys = state.output?.citation_keys ?? [];
    const runPath = result.agent_run_path;
    const querySource = stringValue(retrievalPlan.source) || 'unknown';
    const planReason = stringValue(retrievalPlan.reason) || '无';
    const modelName = stringValue(model.model) || 'unknown';
    const llmStatus = `${stringValue(model.provider) || 'provider?'} / ${modelName} / key=${stringValue(model.api_key) || 'unknown'}`;
    const graphStatus = [
        `${graphNodes.length} nodes`,
        `${graphEdges.length} edges`,
        `${checkpointAfter.length} checkpoints`,
        stringValue(graph.runtime) || undefined,
    ]
        .filter(Boolean)
        .join(' · ');
    const contextStatus = [
        `${contextEngineering.total_calls ?? 0} LLM context calls`,
        `${contextEngineering.estimated_tokens_before ?? 0} -> ${contextEngineering.estimated_tokens_after ?? 0} estimated tokens`,
    ].join(' · ');
    const toolStatus = [
        `${toolReliability.total_calls ?? 0} calls`,
        `${toolReliability.successful_calls ?? 0} success`,
        `${toolReliability.failed_calls ?? 0} failed`,
        `${toolReliability.fallback_calls ?? 0} fallback`,
        ...(toolReliability.error_categories ?? []),
    ].join(' · ');
    const countRows = [
        ['召回', counts.all_recalled_candidates],
        ['去重后', counts.deduped_candidates],
        ['范围过滤后', counts.scoped_candidates],
        ['Evidence shortlist', counts.evidence_shortlist],
        ['已验证', counts.verified_candidates],
        ['最终推荐', counts.final_recommendations],
        ['诊断候选', counts.diagnostic_candidates],
    ]
        .filter(([, value]) => typeof value === 'number')
        .map(([label, value]) => `${label}: ${value}`)
        .join(' · ');
    const openButton = runPath
        ? `<button data-command="openRunRecord">打开 Run Record JSON</button>`
        : '<span class="muted">Run record 未保存。</span>';
    const exportButton = developerMode
        ? `<button data-command="toggleEvalCaseForm">${evalExportExpanded ? '收起 Eval Bad Case' : '导出 Eval Bad Case'}</button>`
        : '';
    const evalForm = developerMode && evalExportExpanded ? renderEvalCaseForm(state.run_id ?? '') : '';
    return `
        <section class="candidate-section agent-state">
            <h2>Agent State</h2>
            <p class="section-note">v17 通过 LangGraph StateGraph 执行推荐流程，并统一记录 Context Budget、工具调用、重试、降级和状态快照。</p>
            <div class="agent-state-grid">
                <div class="state-box">
                    <strong>Run</strong>
                    <code>${htmlEscape(state.run_id ?? 'unknown')}</code>
                </div>
                <div class="state-box">
                    <strong>LLM</strong>
                    <span>${htmlEscape(llmStatus)}</span>
                </div>
                <div class="state-box">
                    <strong>Query Planner</strong>
                    <span>${htmlEscape(querySource)}</span>
                </div>
                <div class="state-box">
                    <strong>Graph</strong>
                    <span>${htmlEscape(graphStatus || '无 graph 元数据')}</span>
                </div>
                <div class="state-box">
                    <strong>Context Budget</strong>
                    <span>${htmlEscape(contextStatus)}</span>
                </div>
                <div class="state-box">
                    <strong>Tool Reliability</strong>
                    <span>${htmlEscape(toolStatus)}</span>
                </div>
                <div class="state-box">
                    <strong>候选流转</strong>
                    <span>${htmlEscape(countRows || '无候选计数')}</span>
                </div>
            </div>
            <p class="section-note">Plan reason: ${htmlEscape(planReason)}</p>
            <p class="section-note">Citation keys: ${htmlEscape(citationKeys.length > 0 ? citationKeys.map((key) => `@${key}`).join(', ') : '无')}</p>
            <div class="toolbar-actions">${openButton}${exportButton}</div>
            ${evalForm}
        </section>
    `;
}

function stringValue(value: unknown): string {
    return typeof value === 'string' ? value : '';
}

function renderEvalCaseForm(runId: string, showCancel = true): string {
    const defaultId = `badcase_${safeEvalIdForRender(runId).slice(0, 16)}`;
    const defaultNotes = `Exported from EasyCite run ${runId || 'unknown'}.`;
    const cancelButton = showCancel ? '<button type="button" data-command="toggleEvalCaseForm">取消</button>' : '';
    return `
        <form id="eval-case-form" class="eval-form">
            <p class="section-note">把这次运行保存为回归测试样本。多篇论文标题用分号分隔。</p>
            <div class="eval-form-grid">
                <label>
                    Case ID
                    <input name="id" value="${htmlEscape(defaultId)}" />
                </label>
                <label>
                    Gold Papers
                    <textarea name="gold" placeholder="最应该推荐的论文标题"></textarea>
                </label>
                <label>
                    Acceptable Papers
                    <textarea name="acceptable" placeholder="也可以接受的论文标题"></textarea>
                </label>
                <label>
                    Forbidden Papers
                    <textarea name="forbidden" placeholder="不应该推荐的论文或类别描述"></textarea>
                </label>
            </div>
            <label>
                Notes
                <textarea name="notes">${htmlEscape(defaultNotes)}</textarea>
            </label>
            <div class="eval-form-actions">
                <button type="button" data-command="saveEvalCase">保存为 Eval Case</button>
                ${cancelButton}
            </div>
        </form>
    `;
}

function safeEvalIdForRender(value: string): string {
    const normalized = value
        .trim()
        .replace(/[^a-zA-Z0-9_-]+/g, '_')
        .replace(/^_+|_+$/g, '');
    return normalized || Date.now().toString(36);
}

function renderRunTraceSection(traceEvents: AuditClaimResult['trace_events']): string {
    if (!traceEvents || traceEvents.length === 0) {
        return '';
    }
    const rows = traceEvents
        .map((event) => {
            const latency =
                event.latency_ms === null || event.latency_ms === undefined ? '' : `${event.latency_ms.toFixed(0)} ms`;
            const before = event.candidate_count_before ?? '';
            const after = event.candidate_count_after ?? '';
            return `
                <tr>
                    <td><code>${htmlEscape(event.stage)}</code></td>
                    <td>${htmlEscape(event.message)}</td>
                    <td>${htmlEscape(String(before))}</td>
                    <td>${htmlEscape(String(after))}</td>
                    <td>${htmlEscape(latency)}</td>
                </tr>
            `;
        })
        .join('');
    const runId = traceEvents[0]?.run_id ?? '';
    return `
        <section class="candidate-section run-trace">
            <h2>运行 Trace</h2>
            <p class="section-note">用于观察本次 Agent workflow 每个阶段的输入输出规模、耗时和决策摘要。Run ID: <code>${htmlEscape(runId)}</code></p>
            <table>
                <thead>
                    <tr>
                        <th>阶段</th>
                        <th>决策摘要</th>
                        <th>Before</th>
                        <th>After</th>
                        <th>耗时</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </section>
    `;
}

function renderTargetTraceSection(trace: string[]): string {
    if (trace.length === 0) {
        return '';
    }
    const lines = trace.map((line) => `<li>${htmlEscape(line)}</li>`).join('');
    return `
        <section class="candidate-section target-trace">
            <h2>目标文献追踪</h2>
            <p class="section-note">用于排查已知论文在召回、过滤、排序、证据验证和最终推荐中的流转。</p>
            <ol>${lines}</ol>
        </section>
    `;
}

function renderStandaloneToolHtml(webview: vscode.Webview, title: string, body: string, scriptBody = ''): string {
    const nonce = Date.now().toString(36);
    const script = scriptBody ? `<script nonce="${nonce}">${scriptBody}</script>` : '';
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${htmlEscape(title)}</title>
    <style>
        body {
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            margin: 0;
            padding: 18px;
        }
        h1 {
            font-size: 20px;
            margin: 0 0 12px;
        }
        h2 {
            border-top: 1px solid var(--vscode-panel-border);
            font-size: 16px;
            margin: 20px 0 8px;
            padding-top: 16px;
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
        .section-note,
        .muted {
            color: var(--vscode-descriptionForeground);
        }
        .toolbar-actions,
        .eval-form-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .agent-state-grid {
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            margin: 10px 0;
        }
        .state-box,
        .eval-form,
        .target-trace ol {
            background: var(--vscode-editorWidget-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            padding: 10px;
        }
        .state-box strong {
            display: block;
            margin-bottom: 4px;
        }
        code,
        .state-box code {
            overflow-wrap: anywhere;
        }
        .eval-form-grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        }
        .eval-form label {
            color: var(--vscode-descriptionForeground);
            display: flex;
            flex-direction: column;
            gap: 5px;
            margin-top: 10px;
        }
        .eval-form input,
        .eval-form textarea {
            background: var(--vscode-input-background);
            border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
            color: var(--vscode-input-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            padding: 6px 8px;
        }
        .eval-form textarea {
            min-height: 70px;
            resize: vertical;
        }
        .run-trace table {
            border-collapse: collapse;
            margin-top: 10px;
            width: 100%;
        }
        .run-trace th,
        .run-trace td {
            border-bottom: 1px solid var(--vscode-panel-border);
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
        }
        .run-trace th {
            color: var(--vscode-descriptionForeground);
            font-weight: 600;
        }
        .target-trace li {
            line-height: 1.5;
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <h1>${htmlEscape(title)}</h1>
    ${body}
    ${script}
</body>
</html>`;
}

function renderEvalCaseToolHtml(webview: vscode.Webview, session: CandidateReviewSession): string {
    const state = session.result.agent_state;
    const runId = state?.run_id ?? session.result.trace_events?.[0]?.run_id ?? '';
    const claim = state?.request?.claim || session.claim;
    const body = `
        <p class="section-note">把当前最近一次候选审核保存为回归测试样本。多篇论文标题用分号或换行分隔。</p>
        <blockquote>${htmlEscape(claim)}</blockquote>
        ${renderEvalCaseForm(runId, false)}
    `;
    return renderStandaloneToolHtml(
        webview,
        'EasyCite Eval Bad Case',
        body,
        `
        const vscode = acquireVsCodeApi();
        document.addEventListener('click', (event) => {
            const button = event.target.closest('button[data-command]');
            if (!button || button.disabled) {
                return;
            }
            if (button.getAttribute('data-command') !== 'saveEvalCase') {
                return;
            }
            const form = document.querySelector('#eval-case-form');
            const data = form ? Object.fromEntries(new FormData(form).entries()) : {};
            vscode.postMessage({ command: 'saveEvalCase', evalCase: data });
        });
        `,
    );
}

function renderEmptyCandidateState(
    session: CandidateReviewSession,
    zoteroEnabled: boolean,
    zoteroEndpoint: string,
): string {
    const summary = renderSummary(session.result);
    const zoteroStatus = zoteroEnabled ? `已启用，endpoint=${zoteroEndpoint || '未配置'}` : '未启用';
    const diagnosticCount = session.result.diagnostic_suggestions?.length ?? 0;
    const heading = diagnosticCount > 0 ? '没有高可信主推荐' : '没有返回可插入的候选文献';
    const tips =
        diagnosticCount > 0
            ? 'EasyCite 已召回候选，但证据判断或范围校验没有让它们进入主推荐区；请查看下方诊断候选了解原因。'
            : zoteroEnabled
              ? '请检查 Zotero 是否正在运行、Better BibTeX/Zotero API 是否可访问，以及当前工作区 settings.json 是否配置了正确 endpoint。'
              : '当前工作区没有启用 Zotero。请在本工作区 settings.json 中设置 citation-agent.zoteroEnabled 和 citation-agent.zoteroEndpoint。';

    return `
        <section class="empty-state">
            <h2>${htmlEscape(heading)}</h2>
            <p>${htmlEscape(summary)}</p>
            <dl>
                <dt>Zotero</dt>
                <dd>${htmlEscape(zoteroStatus)}</dd>
                <dt>推荐候选</dt>
                <dd>${session.result.suggestions.length}</dd>
                <dt>诊断候选</dt>
                <dd>${diagnosticCount}</dd>
            </dl>
            <p class="section-note">${htmlEscape(tips)}</p>
        </section>
    `;
}

function markdownQuote(text: string): string {
    return text
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n')
        .split('\n')
        .map((line) => `> ${line}`)
        .join('\n');
}

function renderBilingualQuote(english: string, chinese?: string): string[] {
    const lines = ['**English**', '', markdownQuote(english), ''];
    if (chinese) {
        lines.push('**中文对照**', '', markdownQuote(chinese), '');
    }
    return lines;
}

function translationMap(result?: TranslateEvidenceResult): Map<string, string> {
    const map = new Map<string, string>();
    for (const item of result?.translations ?? []) {
        if (item.id && item.zh) {
            map.set(item.id, item.zh);
        }
    }
    return map;
}

function collectTranslationTexts(suggestion: CitationSuggestion, claim: string): TranslateEvidenceText[] {
    const passages = suggestion.evidence_passages.length > 0 ? suggestion.evidence_passages : [suggestion.evidence];
    const texts: TranslateEvidenceText[] = [
        { id: 'claim', text: claim },
        { id: 'reason', text: suggestion.reason },
    ];

    passages.forEach((passage, index) => {
        if (passage.text) {
            texts.push({ id: `passage-${index}`, text: passage.text });
        }
    });

    return texts.filter((item) => item.text.trim().length > 0);
}

function renderFullEvidenceMarkdown(
    suggestion: CitationSuggestion,
    claim: string,
    translationResult?: TranslateEvidenceResult,
): string {
    const passages = suggestion.evidence_passages.length > 0 ? suggestion.evidence_passages : [suggestion.evidence];
    const verdict = renderSuggestionDescription(suggestion);
    const authors = suggestion.authors.length > 0 ? suggestion.authors.join(', ') : '未知作者';
    const translations = translationMap(translationResult);
    const translationNotice = translationResult
        ? `中文对照：${translationResult.available ? '已生成' : translationResult.reason}`
        : '中文对照：未请求生成';
    const requestedTranslations = collectTranslationTexts(suggestion, claim).length;
    const metadata = [
        `- Verdict: ${verdict}`,
        `- ${translationNotice}`,
        `- Translation texts requested: ${requestedTranslations}`,
        translationResult ? `- Translation texts returned: ${translationResult.translations.length}` : undefined,
        `- Citation key: ${suggestion.citation_key ? `@${suggestion.citation_key}` : '无引用键'}`,
        `- Authors: ${authors}`,
        suggestion.year ? `- Year: ${suggestion.year}` : undefined,
        suggestion.venue ? `- Venue: ${suggestion.venue}` : undefined,
        suggestion.doi ? `- DOI: ${suggestion.doi}` : undefined,
        suggestion.url ? `- URL: ${suggestion.url}` : undefined,
        suggestion.pdf_path ? `- PDF: ${suggestion.pdf_path}` : undefined,
    ].filter(Boolean);

    const evidenceSections = passages
        .filter((passage) => passage.text)
        .map((passage, index) => {
            const label = index === 0 ? '最相关文段' : `备选文段 ${index + 1}`;
            const location = renderEvidenceLocation(passage);
            const quality = renderEvidenceQualityBadges(passage);
            return [
                `## ${label}`,
                '',
                location ? `_${location}_` : '',
                '',
                quality ? `Evidence quality: ${quality}` : '',
                '',
                ...renderBilingualQuote(passage.text ?? '', translations.get(`passage-${index}`)),
            ].join('\n');
        });

    return [
        '# EasyCite Evidence',
        '',
        '## 待添加引用文段',
        '',
        ...renderBilingualQuote(claim, translations.get('claim')),
        '',
        '## 候选文献',
        '',
        `**${suggestion.title}**`,
        '',
        ...metadata,
        '',
        '## 判断理由',
        '',
        ...renderBilingualQuote(suggestion.reason, translations.get('reason')),
        '',
        ...evidenceSections,
        '',
    ].join('\n');
}

async function translateEvidenceForReview(
    client: LanguageClient,
    suggestion: CitationSuggestion,
    claim: string,
): Promise<TranslateEvidenceResult | undefined> {
    const texts = collectTranslationTexts(suggestion, claim);
    traceInfo(`EasyCite translation request text_count=${texts.length}`);
    return vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'EasyCite: 正在生成中文对照',
            cancellable: false,
        },
        async () => {
            try {
                return await withTimeout(
                    client.sendRequest(TranslateEvidenceRequest, {
                        texts,
                    }),
                    TRANSLATION_REQUEST_TIMEOUT_MS,
                );
            } catch (error) {
                traceError(`EasyCite evidence translation failed: ${error}`);
                return {
                    available: false,
                    reason: `中文对照生成失败：${error}`,
                    translations: [],
                };
            }
        },
    );
}

async function showFullEvidenceDocument(
    client: LanguageClient,
    suggestion: CitationSuggestion,
    claim: string,
): Promise<void> {
    const translationResult = await translateEvidenceForReview(client, suggestion, claim);
    const document = await vscode.workspace.openTextDocument({
        language: 'markdown',
        content: renderFullEvidenceMarkdown(suggestion, claim, translationResult),
    });
    await vscode.window.showTextDocument(document, {
        preview: false,
        viewColumn: vscode.ViewColumn.Beside,
    });
}

interface BibFileLocation {
    uri: vscode.Uri;
    exists: boolean;
    reason: string;
}

interface BibResourceEdit {
    uri: vscode.Uri;
    edit: vscode.TextEdit;
}

interface CitationApplyPlan {
    citationKeys: string[];
    citationEdit: vscode.TextEdit;
    texProject: TexProjectContext;
    bibResourceEdit?: BibResourceEdit;
    bibFile?: BibFileLocation;
    existingBibKeys: Set<string>;
    missingBibKeys: string[];
    resolvedBibtexEntries: ResolvedBibtexEntry[];
    unresolvedBibKeys: Array<{ citation_key: string; reason: string }>;
    preflightChecks: CitationApplyCheck[];
}

interface CitationApplyCheck {
    level: 'ok' | 'warning' | 'error';
    message: string;
}

interface BibtexDocumentValidation {
    keys: Set<string>;
    duplicates: string[];
    malformedEntries: string[];
    unbalancedEntries: string[];
}

function uniqueCitationKeys(citationKeys: string[]): string[] {
    const seen = new Set<string>();
    const keys: string[] = [];
    for (const key of citationKeys) {
        const normalized = key.trim();
        if (!normalized || seen.has(normalized)) {
            continue;
        }
        seen.add(normalized);
        keys.push(normalized);
    }
    return keys;
}

function getCitationCommand(selectedText: string, citationKeys: string[]): string {
    const trailingPunctuation = selectedText.match(/([.!?。！？])([ \t]*)$/);
    const prefix = trailingPunctuation && /[。！？]/.test(trailingPunctuation[1]) ? '\\cite' : '~\\cite';
    return `${prefix}{${citationKeys.join(',')}}`;
}

function getCitationEdit(
    document: vscode.TextDocument,
    selection: vscode.Selection,
    citationKeys: string[],
): vscode.TextEdit {
    const selectedText = document.getText(selection);
    const existingCiteMatch = selectedText.match(
        /(~?\\(?:[a-zA-Z]*cite[a-zA-Z]*|autocite|parencite|textcite)\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\})([.!?。！？])?([ \t]*)$/,
    );
    if (existingCiteMatch) {
        const existingKeys = existingCiteMatch[2]
            .split(',')
            .map((key) => key.trim())
            .filter(Boolean);
        const mergedKeys = uniqueCitationKeys([...existingKeys, ...citationKeys]);
        const suffix = `${getCitationCommand(selectedText, mergedKeys)}${existingCiteMatch[3] ?? ''}${existingCiteMatch[4] ?? ''}`;
        const replaceStart = document.positionAt(document.offsetAt(selection.end) - existingCiteMatch[0].length);
        return vscode.TextEdit.replace(new vscode.Range(replaceStart, selection.end), suffix);
    }

    const trailingMatch = selectedText.match(/([.!?。！？])([ \t]*)$/);
    const citeCommand = getCitationCommand(selectedText, citationKeys);
    if (!trailingMatch) {
        return vscode.TextEdit.insert(selection.end, citeCommand);
    }

    const suffixLength = trailingMatch[1].length + (trailingMatch[2] ?? '').length;
    const replaceStart = document.positionAt(document.offsetAt(selection.end) - suffixLength);
    const replaceRange = new vscode.Range(replaceStart, selection.end);
    return vscode.TextEdit.replace(replaceRange, `${citeCommand}${trailingMatch[1]}${trailingMatch[2] ?? ''}`);
}

function allSuggestions(result: AuditClaimResult): CitationSuggestion[] {
    return [...result.suggestions, ...(result.diagnostic_suggestions ?? [])];
}

function suggestionsForCitationKeys(suggestions: CitationSuggestion[], citationKeys: string[]): CitationSuggestion[] {
    const byKey = new Map<string, CitationSuggestion>();
    for (const suggestion of suggestions) {
        const key = suggestion.citation_key?.trim();
        if (key && !byKey.has(key)) {
            byKey.set(key, suggestion);
        }
    }
    return citationKeys
        .map((key) => byKey.get(key))
        .filter((suggestion): suggestion is CitationSuggestion => Boolean(suggestion));
}

async function buildCitationApplyPlan(
    client: LanguageClient,
    document: vscode.TextDocument,
    selection: vscode.Selection,
    citationKeys: string[],
    suggestions: CitationSuggestion[],
): Promise<CitationApplyPlan> {
    const normalizedKeys = uniqueCitationKeys(citationKeys);
    const citationEdit = getCitationEdit(document, selection, normalizedKeys);
    const texProject = await analyzeTexProject(document);
    const bibFile = await locateProjectBibFile(document, texProject);
    const bibResourceEdit = bibFile ? await getBibResourceEdit(texProject, bibFile.uri.fsPath) : undefined;
    const existingBibKeys = bibFile ? await readBibKeys(bibFile.uri) : new Set<string>();
    const missingBibKeys = bibFile ? normalizedKeys.filter((key) => !existingBibKeys.has(key)) : [];
    let resolvedBibtexEntries: ResolvedBibtexEntry[] = [];
    let unresolvedBibKeys: Array<{ citation_key: string; reason: string }> = [];

    if (missingBibKeys.length > 0) {
        try {
            const result = await withTimeout(
                client.sendRequest(ResolveBibtexRequest, {
                    citation_keys: missingBibKeys,
                    suggestions: suggestionsForCitationKeys(suggestions, missingBibKeys),
                }),
                RESOLVE_BIBTEX_REQUEST_TIMEOUT_MS,
            );
            resolvedBibtexEntries = result.entries ?? [];
            unresolvedBibKeys = result.missing ?? [];
            const resolvedOrMissing = new Set([
                ...resolvedBibtexEntries.map((entry) => entry.citation_key),
                ...unresolvedBibKeys.map((entry) => entry.citation_key),
            ]);
            unresolvedBibKeys.push(
                ...missingBibKeys
                    .filter((key) => !resolvedOrMissing.has(key))
                    .map((key) => ({
                        citation_key: key,
                        reason: 'BibTeX 解析未返回该 key 的条目或失败原因。',
                    })),
            );
        } catch (error) {
            traceError(`EasyCite resolve BibTeX failed: ${error}`);
            unresolvedBibKeys = missingBibKeys.map((key) => ({
                citation_key: key,
                reason: `BibTeX 导出请求失败：${error}`,
            }));
        }
    }

    const preflightChecks = await buildCitationPreflightChecks(
        texProject,
        bibFile,
        bibResourceEdit,
        normalizedKeys,
        missingBibKeys,
        resolvedBibtexEntries,
        unresolvedBibKeys,
    );

    return {
        citationKeys: normalizedKeys,
        citationEdit,
        texProject,
        bibResourceEdit,
        bibFile,
        existingBibKeys,
        missingBibKeys,
        resolvedBibtexEntries,
        unresolvedBibKeys,
        preflightChecks,
    };
}

async function buildCitationPreflightChecks(
    texProject: TexProjectContext,
    bibFile: BibFileLocation | undefined,
    bibResourceEdit: BibResourceEdit | undefined,
    citationKeys: string[],
    missingBibKeys: string[],
    resolvedBibtexEntries: ResolvedBibtexEntry[],
    unresolvedBibKeys: Array<{ citation_key: string; reason: string }>,
): Promise<CitationApplyCheck[]> {
    const checks: CitationApplyCheck[] = [];

    if (citationKeys.length === 0) {
        checks.push({ level: 'error', message: '没有可插入的 citation key。' });
    } else {
        checks.push({ level: 'ok', message: `将插入 ${citationKeys.length} 个 citation key。` });
    }

    if (!bibFile) {
        checks.push({ level: 'warning', message: '未识别到可写入的 .bib 文件，只能插入 cite。' });
        return checks;
    }

    const writableCheck = await checkBibFileWritable(bibFile);
    checks.push(writableCheck);

    if (bibResourceEdit) {
        checks.push({
            level: 'warning',
            message: `TeX 工程 bibliography 声明会在 ${texProject.bibliographyOwner.fsPath} 中同步指向 ${toPosixPath(path.relative(path.dirname(texProject.bibliographyOwner.fsPath), bibFile.uri.fsPath) || path.basename(bibFile.uri.fsPath))}。`,
        });
    } else {
        checks.push({ level: 'ok', message: 'TeX 工程已引用目标 .bib 文件。' });
    }

    if (unresolvedBibKeys.length > 0) {
        for (const item of unresolvedBibKeys) {
            checks.push({
                level: 'error',
                message: `无法补全 @${item.citation_key}: ${item.reason}`,
            });
        }
    }

    if (missingBibKeys.length === 0) {
        checks.push({ level: 'ok', message: '所有 citation key 已存在于目标 .bib。' });
    } else if (resolvedBibtexEntries.length > 0) {
        checks.push({ level: 'ok', message: `已准备追加 ${resolvedBibtexEntries.length} 条 BibTeX。` });
    }

    const duplicateResolvedKeys = duplicateValues(resolvedBibtexEntries.map((entry) => entry.citation_key));
    for (const key of duplicateResolvedKeys) {
        checks.push({ level: 'error', message: `写入计划中重复解析了 @${key}。` });
    }

    for (const entry of resolvedBibtexEntries) {
        const parsedKey = parseBibtexEntryKey(entry.bibtex);
        if (!parsedKey) {
            checks.push({ level: 'error', message: `@${entry.citation_key} 的 BibTeX 条目无法解析入口 key。` });
            continue;
        }
        if (parsedKey !== entry.citation_key) {
            checks.push({
                level: 'error',
                message: `@${entry.citation_key} 的 BibTeX 实际入口 key 是 @${parsedKey}，两者不一致。`,
            });
        }
        if (!hasBalancedBibtexBraces(entry.bibtex)) {
            checks.push({
                level: 'warning',
                message: `@${entry.citation_key} 的 BibTeX 大括号可能不平衡，请写入后人工核查。`,
            });
        }
        const sourceTrust = bibtexSourceTrust(entry);
        if (sourceTrust === 'high') {
            checks.push({ level: 'ok', message: `@${entry.citation_key} BibTeX 来源可信：${entry.source}。` });
        } else if (sourceTrust === 'medium') {
            checks.push({
                level: 'warning',
                message: `@${entry.citation_key} BibTeX 来源为 ${entry.source}，建议人工核对。`,
            });
        } else {
            checks.push({
                level: 'warning',
                message: `@${entry.citation_key} BibTeX 来自低可信兜底来源 ${entry.source}，提交论文前必须人工核对。`,
            });
        }
        for (const warning of entry.quality_warnings ?? []) {
            checks.push({ level: 'warning', message: `@${entry.citation_key}: ${warning}` });
        }
    }

    return checks;
}

async function checkBibFileWritable(bibFile: BibFileLocation): Promise<CitationApplyCheck> {
    try {
        if (bibFile.exists) {
            await fs.promises.access(bibFile.uri.fsPath, fs.constants.W_OK);
            return { level: 'ok', message: `目标 .bib 可写：${bibFile.uri.fsPath}` };
        }
        await fs.promises.mkdir(path.dirname(bibFile.uri.fsPath), { recursive: true });
        await fs.promises.access(path.dirname(bibFile.uri.fsPath), fs.constants.W_OK);
        return { level: 'ok', message: `可创建新的 .bib：${bibFile.uri.fsPath}` };
    } catch (error) {
        return { level: 'error', message: `目标 .bib 不可写：${error}` };
    }
}

function duplicateValues(values: string[]): string[] {
    const seen = new Set<string>();
    const duplicates = new Set<string>();
    for (const value of values) {
        if (seen.has(value)) {
            duplicates.add(value);
        }
        seen.add(value);
    }
    return [...duplicates];
}

function parseBibtexEntryKey(bibtex: string): string | undefined {
    return bibtex.match(/@\w+\s*[{(]\s*([^,\s]+)\s*,/)?.[1]?.trim();
}

function bibtexSourceTrust(entry: ResolvedBibtexEntry): 'high' | 'medium' | 'low' {
    if (entry.source_trust === 'high' || entry.source_trust === 'medium' || entry.source_trust === 'low') {
        return entry.source_trust;
    }
    if (entry.source === 'better-bibtex:zotero') {
        return 'high';
    }
    if (
        entry.source.startsWith('zotero-metadata:') ||
        entry.source.startsWith('doi-metadata:') ||
        entry.source.startsWith('arxiv-metadata:')
    ) {
        return 'medium';
    }
    return 'low';
}

function renderBibtexSourceLabel(entry: ResolvedBibtexEntry): string {
    const trust = bibtexSourceTrust(entry);
    const trustLabel = trust === 'high' ? '高可信' : trust === 'medium' ? '中可信' : '低可信';
    return `${entry.source} · ${trustLabel}${entry.source_rank ? ` · rank ${entry.source_rank}` : ''}`;
}

function hasBalancedBibtexBraces(bibtex: string): boolean {
    const openIndex = findBibtexEntryOpen(bibtex, 0);
    if (openIndex === -1) {
        return false;
    }
    return findMatchingBibtexDelimiter(bibtex, openIndex) !== -1;
}

function findBibtexEntryOpen(content: string, atIndex: number): number {
    const braceIndex = content.indexOf('{', atIndex);
    const parenIndex = content.indexOf('(', atIndex);
    if (braceIndex === -1) {
        return parenIndex;
    }
    if (parenIndex === -1) {
        return braceIndex;
    }
    return Math.min(braceIndex, parenIndex);
}

function findMatchingBibtexDelimiter(content: string, openIndex: number): number {
    const open = content[openIndex];
    const close = open === '(' ? ')' : '}';
    let depth = 0;
    let escaped = false;
    for (let index = openIndex; index < content.length; index += 1) {
        const char = content[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (char === '\\') {
            escaped = true;
            continue;
        }
        if (char === open) {
            depth += 1;
        } else if (char === close) {
            depth -= 1;
            if (depth < 0) {
                return -1;
            }
            if (depth === 0) {
                return index;
            }
        }
    }
    return -1;
}

async function applyCitationPlan(
    editor: vscode.TextEditor,
    plan: CitationApplyPlan,
    mode: 'full' | 'citeOnly',
): Promise<void> {
    const edit = new vscode.WorkspaceEdit();
    const citationEdit = plan.citationEdit;
    if (citationEdit.newText.length > 0 && citationEdit.range.isEmpty) {
        edit.insert(editor.document.uri, citationEdit.range.start, citationEdit.newText);
    } else {
        edit.replace(editor.document.uri, citationEdit.range, citationEdit.newText);
    }
    if (mode === 'full' && plan.bibResourceEdit) {
        if (plan.bibResourceEdit.edit.range.isEmpty) {
            edit.insert(
                plan.bibResourceEdit.uri,
                plan.bibResourceEdit.edit.range.start,
                plan.bibResourceEdit.edit.newText,
            );
        } else {
            edit.replace(plan.bibResourceEdit.uri, plan.bibResourceEdit.edit.range, plan.bibResourceEdit.edit.newText);
        }
    }

    if (mode === 'full' && plan.bibFile && plan.resolvedBibtexEntries.length > 0) {
        const appendText = await buildBibtexAppendText(plan.bibFile, plan.resolvedBibtexEntries);
        if (!plan.bibFile.exists) {
            edit.createFile(plan.bibFile.uri, { ignoreIfExists: true });
        }
        edit.insert(plan.bibFile.uri, await bibAppendPosition(plan.bibFile), appendText);
    }

    const applied = await vscode.workspace.applyEdit(edit);
    if (!applied) {
        throw new Error('WorkspaceEdit was rejected by VS Code.');
    }
}

async function confirmAndApplyCitationPlan(
    client: LanguageClient,
    editor: vscode.TextEditor,
    selection: vscode.Selection,
    citationKeys: string[],
    suggestions: CitationSuggestion[],
): Promise<boolean> {
    const plan = await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'EasyCite: 正在生成引用写入计划',
            cancellable: false,
        },
        async () => buildCitationApplyPlan(client, editor.document, selection, citationKeys, suggestions),
    );

    const picked = await showCitationApplyPlanPanel(editor.document, plan);
    if (!picked) {
        return false;
    }

    await applyCitationPlan(editor, plan, picked);
    const verificationChecks = await verifyCitationApplication(editor.document.uri, plan, picked);
    await showCitationVerificationSummary(plan, picked, verificationChecks);

    const entriesToAppend = plan.resolvedBibtexEntries.length;
    const bibMessage =
        picked === 'citeOnly'
            ? '未更新 BibTeX'
            : entriesToAppend > 0
              ? `已追加 ${entriesToAppend} 条 BibTeX`
              : 'BibTeX 已存在';
    vscode.window.showInformationMessage(`EasyCite: 已插入 \\cite{${plan.citationKeys.join(',')}}，${bibMessage}。`);
    return true;
}

function canApplyFullCitationPlan(plan: CitationApplyPlan): boolean {
    return (
        Boolean(plan.bibFile) &&
        plan.unresolvedBibKeys.length === 0 &&
        !plan.preflightChecks.some((check) => check.level === 'error')
    );
}

function showCitationApplyPlanPanel(
    document: vscode.TextDocument,
    plan: CitationApplyPlan,
): Promise<'full' | 'citeOnly' | undefined> {
    return new Promise((resolve) => {
        let settled = false;
        const panel = vscode.window.createWebviewPanel(
            'citationAgentApplyPlan',
            'EasyCite Apply Plan',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            },
        );

        const finish = (mode: 'full' | 'citeOnly' | undefined) => {
            if (settled) {
                return;
            }
            settled = true;
            panel.dispose();
            resolve(mode);
        };

        panel.webview.html = renderCitationApplyPlanHtml(panel.webview, document, plan);
        panel.webview.onDidReceiveMessage((message: CitationPlanPanelMessage) => {
            if (message.command === 'applyFull') {
                if (canApplyFullCitationPlan(plan)) {
                    finish('full');
                }
                return;
            }
            if (message.command === 'citeOnly') {
                finish('citeOnly');
                return;
            }
            if (message.command === 'cancel') {
                finish(undefined);
            }
        });
        panel.onDidDispose(() => finish(undefined));
    });
}

function renderCitationApplyPlanHtml(
    webview: vscode.Webview,
    document: vscode.TextDocument,
    plan: CitationApplyPlan,
): string {
    const nonce = Date.now().toString(36);
    const fullEnabled = canApplyFullCitationPlan(plan);
    const checkItems = plan.preflightChecks
        .map(
            (check) =>
                `<li class="${check.level}"><span>${renderCheckIcon(check.level)}</span>${htmlEscape(check.message)}</li>`,
        )
        .join('');
    const resolvedEntries =
        plan.resolvedBibtexEntries.length > 0
            ? plan.resolvedBibtexEntries
                  .map(
                      (entry) => `
            <article class="entry trust-${htmlEscape(bibtexSourceTrust(entry))}">
                <h3>@${htmlEscape(entry.citation_key)} <span>${htmlEscape(renderBibtexSourceLabel(entry))}</span></h3>
                ${(entry.quality_warnings ?? []).map((warning) => `<p class="entry-warning">${htmlEscape(warning)}</p>`).join('')}
                <pre>${htmlEscape(entry.bibtex.trim())}</pre>
            </article>
        `,
                  )
                  .join('')
            : '<p class="muted">没有需要追加的 BibTeX 条目。</p>';
    const unresolved =
        plan.unresolvedBibKeys.length > 0
            ? `<ul class="checks">${plan.unresolvedBibKeys.map((item) => `<li class="error"><span>!</span>@${htmlEscape(item.citation_key)}: ${htmlEscape(item.reason)}</li>`).join('')}</ul>`
            : '<p class="muted">没有未解析的 citation key。</p>';
    const bibliographyUpdate = plan.bibResourceEdit
        ? '<p class="notice">将同步更新当前 TeX 文件中的 bibliography 声明，使其指向目标 .bib。</p>'
        : '<p class="muted">当前 TeX 文件已指向目标 .bib。</p>';

    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EasyCite Apply Plan</title>
    <style>
        body {
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            margin: 0;
            padding: 18px;
        }
        .toolbar {
            position: sticky;
            top: 0;
            z-index: 2;
            background: var(--vscode-editor-background);
            border-bottom: 1px solid var(--vscode-panel-border);
            margin-bottom: 16px;
            padding-bottom: 14px;
        }
        h1 {
            font-size: 18px;
            margin: 0 0 12px;
        }
        h2 {
            border-top: 1px solid var(--vscode-panel-border);
            font-size: 15px;
            margin: 20px 0 8px;
            padding-top: 14px;
        }
        h3 {
            font-size: 14px;
            margin: 0 0 8px;
        }
        h3 span,
        .muted {
            color: var(--vscode-descriptionForeground);
        }
        .grid {
            display: grid;
            gap: 8px;
            grid-template-columns: minmax(120px, 180px) minmax(0, 1fr);
        }
        .label {
            color: var(--vscode-descriptionForeground);
        }
        .value,
        code {
            overflow-wrap: anywhere;
        }
        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }
        button {
            background: var(--vscode-button-background);
            border: 0;
            border-radius: 3px;
            color: var(--vscode-button-foreground);
            cursor: pointer;
            padding: 6px 11px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        button.secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        button.secondary:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }
        button:disabled {
            cursor: not-allowed;
            opacity: 0.45;
        }
        .checks {
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .checks li {
            border-left: 3px solid var(--vscode-panel-border);
            line-height: 1.45;
            margin: 7px 0;
            padding: 6px 8px;
        }
        .checks span {
            display: inline-block;
            font-weight: 700;
            margin-right: 7px;
            width: 1em;
        }
        .checks .ok {
            border-color: var(--vscode-testing-iconPassed);
        }
        .checks .warning {
            border-color: var(--vscode-editorWarning-foreground);
        }
        .checks .error {
            border-color: var(--vscode-editorError-foreground);
        }
        .notice {
            border-left: 3px solid var(--vscode-editorWarning-foreground);
            color: var(--vscode-descriptionForeground);
            padding-left: 8px;
        }
        .entry {
            border: 1px solid var(--vscode-panel-border);
            border-radius: 6px;
            margin-bottom: 12px;
            padding: 11px;
        }
        .entry.trust-high {
            border-left: 3px solid var(--vscode-testing-iconPassed);
        }
        .entry.trust-medium {
            border-left: 3px solid var(--vscode-editorWarning-foreground);
        }
        .entry.trust-low {
            border-left: 3px solid var(--vscode-editorError-foreground);
        }
        .entry-warning {
            color: var(--vscode-editorWarning-foreground);
            margin: 4px 0 8px;
        }
        pre {
            background: var(--vscode-textCodeBlock-background);
            overflow-x: auto;
            padding: 10px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <section class="toolbar">
        <h1>EasyCite Apply Plan</h1>
        <div class="grid">
            <div class="label">TeX</div>
            <div class="value">${htmlEscape(document.uri.fsPath)}</div>
            <div class="label">主 TeX</div>
            <div class="value">${htmlEscape(plan.texProject.rootFile)} · ${htmlEscape(plan.texProject.rootReason)}</div>
            <div class="label">插入</div>
            <div class="value"><code>\\cite{${htmlEscape(plan.citationKeys.join(','))}}</code></div>
            <div class="label">BibTeX</div>
            <div class="value">${plan.bibFile ? htmlEscape(plan.bibFile.uri.fsPath) : '未识别'}</div>
            <div class="label">来源</div>
            <div class="value">${plan.bibFile ? htmlEscape(plan.bibFile.reason) : '无'}</div>
        </div>
        <div class="actions">
            <button ${fullEnabled ? '' : 'disabled'} data-command="applyFull">插入引用并更新 BibTeX</button>
            <button class="secondary" data-command="citeOnly">仅插入 cite</button>
            <button class="secondary" data-command="cancel">取消</button>
        </div>
    </section>
    <main>
        <h2>写入前预检</h2>
        <ul class="checks">${checkItems}</ul>
        <h2>TeX bibliography 声明</h2>
        ${bibliographyUpdate}
        <h2>未解析条目</h2>
        ${unresolved}
        <h2>将追加的 BibTeX</h2>
        ${resolvedEntries}
    </main>
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

function renderCheckIcon(level: CitationApplyCheck['level']): string {
    if (level === 'ok') {
        return '✓';
    }
    if (level === 'warning') {
        return '!';
    }
    return '×';
}

async function verifyCitationApplication(
    documentUri: vscode.Uri,
    plan: CitationApplyPlan,
    mode: 'full' | 'citeOnly',
): Promise<CitationApplyCheck[]> {
    const checks: CitationApplyCheck[] = [];
    const document = await vscode.workspace.openTextDocument(documentUri);
    const texContent = document.getText();
    const insertedKeys = new Set(extractExistingCitations(texContent));
    const missingCitationKeys = plan.citationKeys.filter((key) => !insertedKeys.has(key));
    if (missingCitationKeys.length === 0) {
        checks.push({ level: 'ok', message: 'TeX 中已检测到插入的 citation key。' });
    } else {
        checks.push({
            level: 'error',
            message: `TeX 中未检测到这些 citation key: ${missingCitationKeys.map((key) => `@${key}`).join(', ')}。`,
        });
    }

    if (mode === 'citeOnly') {
        checks.push({ level: 'warning', message: '本次选择了仅插入 cite，未校验 BibTeX 写入。' });
        return checks;
    }

    if (!plan.bibFile) {
        checks.push({ level: 'error', message: '写入后校验失败：写入计划没有目标 .bib 文件。' });
        return checks;
    }

    const bibliographyDocument = await vscode.workspace.openTextDocument(plan.texProject.bibliographyOwner);
    const documentDir = path.dirname(bibliographyDocument.uri.fsPath);
    const referenced = new Set(resolveBibReferences(bibliographyDocument.getText(), documentDir));
    if (referenced.has(plan.bibFile.uri.fsPath)) {
        checks.push({ level: 'ok', message: 'TeX bibliography 声明已指向目标 .bib。' });
    } else {
        checks.push({ level: 'error', message: `TeX bibliography 声明没有指向目标 .bib: ${plan.bibFile.uri.fsPath}` });
    }

    try {
        const bibDocument = await vscode.workspace.openTextDocument(plan.bibFile.uri);
        const validation = validateBibtexDocument(bibDocument.getText());
        const missingBibKeys = plan.citationKeys.filter((key) => !validation.keys.has(key));
        if (missingBibKeys.length === 0) {
            checks.push({ level: 'ok', message: '目标 .bib 中已检测到全部 citation key。' });
        } else {
            checks.push({
                level: 'error',
                message: `目标 .bib 中缺少这些 key: ${missingBibKeys.map((key) => `@${key}`).join(', ')}。`,
            });
        }
        if (validation.duplicates.length === 0) {
            checks.push({ level: 'ok', message: '目标 .bib 未检测到重复 citation key。' });
        } else {
            checks.push({
                level: 'error',
                message: `目标 .bib 存在重复 key: ${validation.duplicates.map((key) => `@${key}`).join(', ')}。`,
            });
        }
        if (validation.malformedEntries.length === 0 && validation.unbalancedEntries.length === 0) {
            checks.push({ level: 'ok', message: '目标 .bib 条目结构和大括号校验通过。' });
        } else {
            for (const item of validation.malformedEntries.slice(0, 5)) {
                checks.push({ level: 'error', message: `目标 .bib 存在无法解析的条目片段：${item}` });
            }
            for (const item of validation.unbalancedEntries.slice(0, 5)) {
                checks.push({ level: 'error', message: `目标 .bib 存在大括号/括号不平衡条目：${item}` });
            }
        }
    } catch (error) {
        checks.push({ level: 'error', message: `无法读取目标 .bib 进行后验校验：${error}` });
    }

    return checks;
}

async function showCitationVerificationSummary(
    plan: CitationApplyPlan,
    mode: 'full' | 'citeOnly',
    checks: CitationApplyCheck[],
): Promise<void> {
    const errors = checks.filter((check) => check.level === 'error');
    const warnings = checks.filter((check) => check.level === 'warning');
    const detail = checks.map((check) => `${renderCheckIcon(check.level)} ${check.message}`).join('\n');
    if (errors.length > 0) {
        await vscode.window.showErrorMessage(
            'EasyCite: 引用已尝试写入，但后验校验发现问题。',
            { modal: true, detail },
            '知道了',
        );
        return;
    }
    if (warnings.length > 0) {
        await vscode.window.showWarningMessage(
            `EasyCite: 已${mode === 'full' ? '写入引用和 BibTeX' : '插入 cite'}，但有提示需要留意。`,
            { modal: true, detail },
            '知道了',
        );
        return;
    }
    traceInfo(`EasyCite post-apply verification passed for ${plan.citationKeys.length} key(s).`);
}

function renderCitationPlanDetail(document: vscode.TextDocument, plan: CitationApplyPlan): string {
    const lines = [`TeX: ${document.uri.fsPath}`, `插入: \\cite{${plan.citationKeys.join(',')}}`];
    if (plan.bibFile) {
        lines.push(
            `BibTeX: ${plan.bibFile.uri.fsPath}`,
            `BibTeX 来源: ${plan.bibFile.reason}`,
            ...(plan.bibResourceEdit ? ['TeX bibliography 声明: 将同步更新'] : []),
            `已存在 key: ${plan.citationKeys.filter((key) => plan.existingBibKeys.has(key)).length}`,
            `待追加条目: ${plan.resolvedBibtexEntries.length}`,
        );
    } else {
        lines.push('BibTeX: 未识别到当前工程的 .bib 文件；只能安全地插入 cite。');
    }
    if (plan.unresolvedBibKeys.length > 0) {
        lines.push(
            '',
            '未能补全 BibTeX:',
            ...plan.unresolvedBibKeys.map((item) => `- @${item.citation_key}: ${item.reason}`),
        );
    }
    return lines.join('\n');
}

async function locateProjectBibFile(
    document: vscode.TextDocument,
    texProject: TexProjectContext,
): Promise<BibFileLocation | undefined> {
    if (document.uri.scheme !== 'file') {
        return undefined;
    }

    const documentDir = path.dirname(document.uri.fsPath);
    const bibliographyDir = path.dirname(texProject.bibliographyOwner.fsPath);
    const referenced = texProject.bibliographyReferences;
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
    const searchPattern = new vscode.RelativePattern(workspaceFolder ?? documentDir, '**/*.bib');
    const found = await vscode.workspace.findFiles(searchPattern, '**/{.git,.venv,node_modules,dist,out}/**', 20);
    const existing = found.filter((uri) => fs.existsSync(uri.fsPath));
    if (existing.length > 0) {
        const referencedSet = new Set(referenced);
        const sorted = existing.sort(
            (first, second) =>
                bibFilePreference(first.fsPath, document.uri.fsPath, referencedSet) -
                bibFilePreference(second.fsPath, document.uri.fsPath, referencedSet),
        );
        const selected = sorted[0];
        return {
            uri: selected,
            exists: true,
            reason: referencedSet.has(selected.fsPath)
                ? `主 TeX bibliography 声明文件 (${path.basename(texProject.bibliographyOwner.fsPath)})`
                : '已存在的项目 .bib 文件',
        };
    }

    if (referenced.length > 0) {
        const first = referenced[0];
        return {
            uri: vscode.Uri.file(first),
            exists: false,
            reason: '当前 TeX 文件中的 bibliography 声明',
        };
    }

    return {
        uri: vscode.Uri.file(path.join(bibliographyDir, 'references.bib')),
        exists: false,
        reason: '未找到已有 .bib，使用默认 references.bib',
    };
}

function bibFilePreference(filePath: string, documentPath: string, referencedPaths: Set<string>): number {
    const documentDir = path.dirname(documentPath);
    const documentBibName = `${path.basename(documentPath, path.extname(documentPath)).toLowerCase()}.bib`;
    const basename = path.basename(filePath).toLowerCase();
    const sameBaseName = basename === documentBibName ? -30 : 0;
    const referenced = referencedPaths.has(filePath) ? -20 : 0;
    const sameDir = path.dirname(filePath) === documentDir ? 0 : 30;
    const nameScore = ['references.bib', 'reference.bib', 'refs.bib', 'main.bib'].includes(basename) ? 0 : 5;
    return sameBaseName + referenced + sameDir + nameScore;
}

async function getBibResourceEdit(
    texProject: TexProjectContext,
    bibPath: string,
): Promise<BibResourceEdit | undefined> {
    const document = await vscode.workspace.openTextDocument(texProject.bibliographyOwner);
    const documentDir = path.dirname(document.uri.fsPath);
    const texContent = document.getText();
    const referenced = new Set(resolveBibReferences(texContent, documentDir));
    if (referenced.has(bibPath)) {
        return undefined;
    }

    const relativePath = toPosixPath(path.relative(documentDir, bibPath) || path.basename(bibPath));
    const addBibResource = /\\addbibresource(?:\[[^\]]*\])?\{([^}]+)\}/g;
    const bibliography = /\\bibliography\{([^}]+)\}/g;
    const addMatch = addBibResource.exec(texContent);
    if (addMatch?.index !== undefined) {
        const start = document.positionAt(addMatch.index + addMatch[0].indexOf('{') + 1);
        const end = document.positionAt(addMatch.index + addMatch[0].lastIndexOf('}'));
        return {
            uri: document.uri,
            edit: vscode.TextEdit.replace(new vscode.Range(start, end), relativePath),
        };
    }

    const bibMatch = bibliography.exec(texContent);
    if (bibMatch?.index !== undefined) {
        const start = document.positionAt(bibMatch.index + bibMatch[0].indexOf('{') + 1);
        const end = document.positionAt(bibMatch.index + bibMatch[0].lastIndexOf('}'));
        return {
            uri: document.uri,
            edit: vscode.TextEdit.replace(new vscode.Range(start, end), relativePath),
        };
    }

    const beginDocument = texContent.search(/\\begin\{document\}/);
    const insertOffset = beginDocument >= 0 ? beginDocument : 0;
    const prefix = insertOffset > 0 && !texContent.slice(0, insertOffset).endsWith('\n') ? '\n' : '';
    return {
        uri: document.uri,
        edit: vscode.TextEdit.insert(document.positionAt(insertOffset), `${prefix}\\addbibresource{${relativePath}}\n`),
    };
}

function toPosixPath(filePath: string): string {
    return filePath.split(path.sep).join('/');
}

async function readBibKeys(uri: vscode.Uri): Promise<Set<string>> {
    try {
        const content = await fs.promises.readFile(uri.fsPath, 'utf-8');
        return validateBibtexDocument(content).keys;
    } catch {
        return new Set<string>();
    }
}

function parseBibKeysFromContent(content: string): Set<string> {
    return validateBibtexDocument(content).keys;
}

function validateBibtexDocument(content: string): BibtexDocumentValidation {
    const keys = new Set<string>();
    const seen = new Set<string>();
    const duplicates = new Set<string>();
    const malformedEntries: string[] = [];
    const unbalancedEntries: string[] = [];
    const entryRegex = /@\w+\s*[{(]\s*([^,\s]+)\s*,/g;
    let match = entryRegex.exec(content);
    while (match) {
        const key = match[1];
        if (seen.has(key)) {
            duplicates.add(key);
        }
        seen.add(key);
        keys.add(key);
        match = entryRegex.exec(content);
    }
    let index = 0;
    while (index < content.length) {
        const atIndex = content.indexOf('@', index);
        if (atIndex === -1) {
            break;
        }
        const openIndex = findBibtexEntryOpen(content, atIndex);
        if (openIndex === -1) {
            malformedEntries.push(content.slice(atIndex, Math.min(atIndex + 80, content.length)).trim());
            index = atIndex + 1;
            continue;
        }
        const closeIndex = findMatchingBibtexDelimiter(content, openIndex);
        if (closeIndex === -1) {
            const key = content
                .slice(openIndex + 1, Math.min(openIndex + 80, content.length))
                .match(/^\s*([^,\s]+)/)?.[1];
            unbalancedEntries.push(
                key ? `@${key}` : content.slice(atIndex, Math.min(atIndex + 80, content.length)).trim(),
            );
            break;
        }
        index = closeIndex + 1;
    }
    return {
        keys,
        duplicates: [...duplicates],
        malformedEntries,
        unbalancedEntries,
    };
}

async function buildBibtexAppendText(bibFile: BibFileLocation, entries: ResolvedBibtexEntry[]): Promise<string> {
    let existing = '';
    if (bibFile.exists) {
        try {
            existing = await fs.promises.readFile(bibFile.uri.fsPath, 'utf-8');
        } catch {
            existing = '';
        }
    }
    const prefix = existing.trim().length === 0 ? '' : existing.endsWith('\n') ? '\n' : '\n\n';
    const body = entries
        .map((entry) => entry.bibtex.trim())
        .filter(Boolean)
        .join('\n\n');
    return `${prefix}${body}\n`;
}

async function bibAppendPosition(bibFile: BibFileLocation): Promise<vscode.Position> {
    if (!bibFile.exists) {
        return new vscode.Position(0, 0);
    }
    const document = await vscode.workspace.openTextDocument(bibFile.uri);
    return document.lineAt(Math.max(document.lineCount - 1, 0)).range.end;
}

function buildCandidateQuickPickItems(
    suggestions: CitationSuggestion[],
    selectedKeys: Set<string>,
): CandidateQuickPickItem[] {
    const items: CandidateQuickPickItem[] = [];

    if (selectedKeys.size > 0) {
        items.push({
            itemType: 'insertSelected',
            label: `$(check) 插入已选择的 ${selectedKeys.size} 条引用`,
            description: [...selectedKeys].map((key) => `@${key}`).join(', '),
            alwaysShow: true,
        });
        items.push({
            itemType: 'clearSelected',
            label: '$(clear-all) 清空已选择引用',
            alwaysShow: true,
        });
    }

    items.push({
        itemType: 'endReview',
        label: '$(x) 结束候选审核',
        alwaysShow: true,
    });

    for (const suggestion of suggestions) {
        const selected = suggestion.citation_key ? selectedKeys.has(suggestion.citation_key) : false;
        items.push({
            itemType: 'candidate',
            label: `${selected ? '$(check) ' : ''}${suggestion.title}`,
            description: renderSuggestionDescription(suggestion),
            detail: renderSuggestionDetail(suggestion),
            suggestion,
        });
    }

    return items;
}

async function showCitationCandidates(
    editor: vscode.TextEditor,
    selection: vscode.Selection,
    result: AuditClaimResult,
    client: LanguageClient,
    selectedKeys: Set<string>,
    controls: CandidateReviewControls,
    getRunningClient?: (preferred?: LanguageClient) => Promise<LanguageClient | undefined>,
): Promise<void> {
    if (result.suggestions.length === 0) {
        vscode.window.showInformationMessage('EasyCite: 当前没有从 .bib/Zotero 找到候选引用。');
        return;
    }

    while (true) {
        const recordFeedback = async (suggestion: CitationSuggestion, action: 'accepted' | 'rejected') => {
            const config = vscode.workspace.getConfiguration('citation-agent');
            if (!config.get<boolean>('memoryEnabled', true)) {
                return;
            }
            await client
                .sendRequest(RecordFeedbackRequest, {
                    workspace_path: getWorkspacePath(editor.document),
                    file_path: editor.document.uri.fsPath,
                    claim: editor.document.getText(selection),
                    context: getParagraphContext(editor.document, selection),
                    action,
                    suggestion,
                })
                .then(
                    () => undefined,
                    (error) => traceError(`EasyCite memory feedback failed: ${error}`),
                );
        };

        const picked = await vscode.window.showQuickPick(
            buildCandidateQuickPickItems(result.suggestions, selectedKeys),
            {
                title: 'EasyCite: 候选引用',
                placeHolder:
                    selectedKeys.size > 0
                        ? '继续查看证据、勾选候选，或选择顶部操作插入已选引用'
                        : '选择一条候选以查看完整证据、加入引用列表或直接插入',
                ignoreFocusOut: false,
                matchOnDescription: true,
                matchOnDetail: true,
            },
        );

        if (!picked) {
            return;
        }

        if (picked.itemType === 'insertSelected') {
            const liveClient = getRunningClient ? await getRunningClient(client) : client;
            if (!liveClient) {
                vscode.window.showErrorMessage('EasyCite: Python 后端尚未启动，无法补全 BibTeX。');
                return;
            }
            const applied = await confirmAndApplyCitationPlan(
                liveClient,
                editor,
                selection,
                [...selectedKeys],
                allSuggestions(result),
            );
            if (applied) {
                controls.endReview();
            }
            return;
        }

        if (picked.itemType === 'clearSelected') {
            selectedKeys.clear();
            continue;
        }

        if (picked.itemType === 'endReview') {
            controls.endReview();
            return;
        }

        const suggestion = picked.suggestion;
        if (!suggestion) {
            continue;
        }

        const citationKey = suggestion.citation_key;
        const selectionAction = citationKey && selectedKeys.has(citationKey) ? '取消选择' : '加入引用列表';
        const actions = citationKey ? ['查看完整证据', selectionAction, '仅插入此条'] : ['查看完整证据'];
        const action = await vscode.window.showInformationMessage(
            `EasyCite: ${suggestion.title}`,
            {
                modal: true,
                detail: '查看完整证据后会回到候选列表；可以选择多条候选，最后一次性插入。',
            },
            ...actions,
        );

        if (action === '查看完整证据') {
            const selectedText = editor.document.getText(selection);
            await showFullEvidenceDocument(client, suggestion, selectedText);
            controls.pauseReview();
            vscode.window.showInformationMessage('EasyCite: 已打开完整证据。点击状态栏“返回候选”可继续比较其他候选。');
            return;
        }

        if (action === '加入引用列表' && citationKey) {
            selectedKeys.add(citationKey);
            await recordFeedback(suggestion, 'accepted');
            continue;
        }

        if (action === '取消选择' && citationKey) {
            selectedKeys.delete(citationKey);
            await recordFeedback(suggestion, 'rejected');
            continue;
        }

        if (action === '仅插入此条' && citationKey) {
            const liveClient = getRunningClient ? await getRunningClient(client) : client;
            if (!liveClient) {
                vscode.window.showErrorMessage('EasyCite: Python 后端尚未启动，无法补全 BibTeX。');
                return;
            }
            const applied = await confirmAndApplyCitationPlan(
                liveClient,
                editor,
                selection,
                [citationKey],
                allSuggestions(result),
            );
            if (applied) {
                await recordFeedback(suggestion, 'accepted');
                controls.endReview();
            }
            return;
        }
    }
}

export function registerAuditSelectionCommand(
    context: vscode.ExtensionContext,
    getClient: ClientProvider,
    ensureServer: () => Promise<LanguageClient | undefined>,
): void {
    let auditInFlight = false;
    let activeReviewSession: CandidateReviewSession | undefined;
    let candidatePanel: vscode.WebviewPanel | undefined;
    const citationDiagnostics = vscode.languages.createDiagnosticCollection(CITATION_DIAGNOSTIC_SOURCE);
    const reviewStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    reviewStatus.command = 'citation-agent.resumeCandidateReview';
    context.subscriptions.push(reviewStatus, citationDiagnostics);

    const scanTimers = new Map<string, NodeJS.Timeout>();
    const scansInFlight = new Set<string>();

    function citationDiagnosticCode(diagnostic: vscode.Diagnostic): string | number | undefined {
        const code = typeof diagnostic.code === 'object' ? diagnostic.code.value : diagnostic.code;
        return code;
    }

    function isCitationDiagnostic(diagnostic: vscode.Diagnostic): boolean {
        const code = citationDiagnosticCode(diagnostic);
        return (
            diagnostic.source === CITATION_DIAGNOSTIC_SOURCE &&
            (code === CITATION_DIAGNOSTIC_CODE || code === CITATION_AUTO_DIAGNOSTIC_CODE)
        );
    }

    function isAutoCitationDiagnostic(diagnostic: vscode.Diagnostic): boolean {
        return (
            diagnostic.source === CITATION_DIAGNOSTIC_SOURCE &&
            citationDiagnosticCode(diagnostic) === CITATION_AUTO_DIAGNOSTIC_CODE
        );
    }

    function isAuditCitationDiagnostic(diagnostic: vscode.Diagnostic): boolean {
        return (
            diagnostic.source === CITATION_DIAGNOSTIC_SOURCE &&
            citationDiagnosticCode(diagnostic) === CITATION_DIAGNOSTIC_CODE
        );
    }

    function upsertCitationDiagnostic(
        uri: vscode.Uri,
        range: vscode.Range,
        result: AuditClaimResult,
        candidateCount: number,
    ): void {
        const existing = citationDiagnostics.get(uri) ?? [];
        const retained = existing.filter(
            (diagnostic) => !(isCitationDiagnostic(diagnostic) && diagnostic.range.isEqual(range)),
        );
        const diagnostic = new vscode.Diagnostic(
            range,
            `需要引用：${result.claim_type}，候选 ${candidateCount} 条。${result.reason}`,
            vscode.DiagnosticSeverity.Warning,
        );
        diagnostic.source = CITATION_DIAGNOSTIC_SOURCE;
        diagnostic.code = CITATION_DIAGNOSTIC_CODE;
        citationDiagnostics.set(uri, [...retained, diagnostic]);
    }

    function clearCitationDiagnostics(uri?: vscode.Uri, range?: vscode.Range): void {
        if (!uri) {
            citationDiagnostics.clear();
            return;
        }
        if (!range) {
            citationDiagnostics.delete(uri);
            return;
        }
        const existing = citationDiagnostics.get(uri) ?? [];
        citationDiagnostics.set(
            uri,
            existing.filter(
                (diagnostic) => !(isCitationDiagnostic(diagnostic) && diagnostic.range.intersection(range)),
            ),
        );
    }

    function setAutoCitationDiagnostics(uri: vscode.Uri, diagnostics: vscode.Diagnostic[]): void {
        const existing = citationDiagnostics.get(uri) ?? [];
        const retained = existing.filter((diagnostic) => !isAutoCitationDiagnostic(diagnostic));
        citationDiagnostics.set(uri, [...retained, ...diagnostics]);
    }

    function isTexDocument(document: vscode.TextDocument): boolean {
        return (
            document.uri.scheme === 'file' &&
            (document.languageId === 'latex' ||
                document.languageId === 'tex' ||
                document.uri.fsPath.toLowerCase().endsWith('.tex'))
        );
    }

    function scheduleTexProjectScan(document: vscode.TextDocument | undefined, reason = 'auto'): void {
        if (!document || !isTexDocument(document)) {
            return;
        }
        const key = vscode.workspace.getWorkspaceFolder(document.uri)?.uri.toString() ?? document.uri.toString();
        const existing = scanTimers.get(key);
        if (existing) {
            clearTimeout(existing);
        }
        scanTimers.set(
            key,
            setTimeout(() => {
                scanTimers.delete(key);
                void scanTexProjectForCitationDiagnostics(document, reason);
            }, TEX_PROJECT_SCAN_DEBOUNCE_MS),
        );
    }

    async function scanTexProjectForCitationDiagnostics(
        entryDocument: vscode.TextDocument,
        reason = 'manual',
    ): Promise<void> {
        if (!isTexDocument(entryDocument)) {
            return;
        }
        const project = await analyzeTexProject(entryDocument);
        const scanKey = project.rootFile;
        if (scansInFlight.has(scanKey)) {
            return;
        }
        scansInFlight.add(scanKey);
        try {
            const files = uniqueCitationKeys([project.rootFile, ...project.includedFiles])
                .filter((filePath) => filePath.toLowerCase().endsWith('.tex'))
                .slice(0, MAX_AUTO_SCAN_FILES);
            for (const filePath of files) {
                const uri = vscode.Uri.file(filePath);
                const document = await vscode.workspace.openTextDocument(uri);
                setAutoCitationDiagnostics(uri, await buildAutoCitationDiagnostics(document));
            }
            traceInfo(
                `EasyCite TeX project scan completed: root=${project.rootFile}, files=${files.length}, reason=${reason}.`,
            );
        } catch (error) {
            traceError(`EasyCite TeX project scan failed: ${error}`);
        } finally {
            scansInFlight.delete(scanKey);
        }
    }

    async function buildAutoCitationDiagnostics(document: vscode.TextDocument): Promise<vscode.Diagnostic[]> {
        const diagnostics: vscode.Diagnostic[] = [];
        const candidates = await refineCitationScanCandidatesWithEmbedding(findCitationCandidateRanges(document));
        for (const candidate of candidates) {
            const diagnostic = new vscode.Diagnostic(
                candidate.range,
                renderAutoCitationDiagnosticMessage(candidate),
                vscode.DiagnosticSeverity.Warning,
            );
            diagnostic.source = CITATION_DIAGNOSTIC_SOURCE;
            diagnostic.code = CITATION_AUTO_DIAGNOSTIC_CODE;
            diagnostics.push(diagnostic);
            if (diagnostics.length >= MAX_AUTO_SCAN_DIAGNOSTICS_PER_FILE) {
                break;
            }
        }
        return diagnostics;
    }

    async function refineCitationScanCandidatesWithEmbedding(
        candidates: CitationScanCandidate[],
    ): Promise<CitationScanCandidate[]> {
        if (candidates.length === 0) {
            return [];
        }

        const config = vscode.workspace.getConfiguration('citation-agent');
        const scanUseEmbedding = config.get<boolean>('scanUseEmbedding', true);
        const embeddingEnabled = config.get<boolean>('embeddingEnabled', false);
        const threshold = config.get<number>('scanEmbeddingThreshold', DEFAULT_SCAN_EMBEDDING_THRESHOLD);
        const maxSentences = config.get<number>('scanMaxSentences', 300);
        const limited = candidates.slice(0, Math.max(maxSentences, 1));
        if (!scanUseEmbedding || !embeddingEnabled) {
            return limited;
        }

        const client = getClient();
        if (!client || client.state !== State.Running) {
            return limited;
        }

        try {
            const sentences: CitationScanSentence[] = limited.map((candidate) => ({
                id: candidate.id,
                text: candidate.text,
            }));
            const result = await withTimeout(
                client.sendRequest(RankCitationScanRequest, { sentences }),
                RANK_CITATION_SCAN_TIMEOUT_MS,
            );
            if (!result.available || result.scores.length === 0) {
                traceInfo(`EasyCite scan embedding fallback used: ${result.reason}`);
                return limited;
            }
            const scoreById = new Map(result.scores.map((score) => [score.id, score]));
            const refined = limited
                .map((candidate) => {
                    const score = scoreById.get(candidate.id);
                    if (!score) {
                        return candidate;
                    }
                    return {
                        ...candidate,
                        lexicalReason: `${candidate.lexicalReason}; embedding=${score.score.toFixed(2)} (${score.matched_prototype})`,
                        lexicalScore: Math.max(candidate.lexicalScore, score.score),
                    };
                })
                .filter((candidate) => {
                    const score = scoreById.get(candidate.id)?.score ?? 0;
                    return score >= threshold || candidate.lexicalScore >= 0.82;
                });
            traceInfo(
                `EasyCite scan embedding applied: input=${limited.length}, kept=${refined.length}, threshold=${threshold}, provider=${result.provider}, model=${result.model_id}.`,
            );
            return refined;
        } catch (error) {
            traceError(`EasyCite scan embedding failed; lexical scan retained: ${error}`);
            return limited;
        }
    }

    function renderAutoCitationDiagnosticMessage(candidate: CitationScanCandidate): string {
        return `疑似需要引用：${candidate.lexicalReason}。点击灯泡让 EasyCite 审计此句并推荐候选文献。`;
    }

    function findCitationCandidateRanges(document: vscode.TextDocument): CitationScanCandidate[] {
        const candidates: CitationScanCandidate[] = [];
        let paragraphStartLine: number | undefined;
        let paragraphEndLine: number | undefined;

        const flushParagraph = () => {
            if (paragraphStartLine === undefined || paragraphEndLine === undefined) {
                return;
            }
            const paragraphRange = new vscode.Range(
                new vscode.Position(paragraphStartLine, 0),
                document.lineAt(paragraphEndLine).range.end,
            );
            const paragraphText = document.getText(paragraphRange);
            candidates.push(...findCitationCandidateSentenceRanges(document, paragraphRange, paragraphText));
            paragraphStartLine = undefined;
            paragraphEndLine = undefined;
        };

        let ignoredEnvironmentDepth = 0;
        for (let line = 0; line < document.lineCount; line += 1) {
            const rawLine = document.lineAt(line).text;
            const lineWithoutComment = stripTexComment(rawLine);
            if (
                /\\begin\{(?:equation|align|figure|table|algorithm|lstlisting|verbatim|tikzpicture)\*?\}/.test(
                    lineWithoutComment,
                )
            ) {
                ignoredEnvironmentDepth += 1;
                flushParagraph();
                continue;
            }
            if (
                /\\end\{(?:equation|align|figure|table|algorithm|lstlisting|verbatim|tikzpicture)\*?\}/.test(
                    lineWithoutComment,
                )
            ) {
                ignoredEnvironmentDepth = Math.max(ignoredEnvironmentDepth - 1, 0);
                flushParagraph();
                continue;
            }
            if (ignoredEnvironmentDepth > 0 || shouldBreakCitationScanParagraph(lineWithoutComment)) {
                flushParagraph();
                continue;
            }
            if (paragraphStartLine === undefined) {
                paragraphStartLine = line;
            }
            paragraphEndLine = line;
        }
        flushParagraph();
        return candidates;
    }

    function findCitationCandidateSentenceRanges(
        document: vscode.TextDocument,
        paragraphRange: vscode.Range,
        paragraphText: string,
    ): CitationScanCandidate[] {
        const candidates: CitationScanCandidate[] = [];
        const sentencePattern = /[^.!?。！？]+[.!?。！？]/g;
        let match = sentencePattern.exec(paragraphText);
        while (match) {
            const sentence = match[0];
            const classification = classifyCitationCandidateSentence(sentence);
            if (classification) {
                const start = document.positionAt(document.offsetAt(paragraphRange.start) + match.index);
                const end = document.positionAt(
                    document.offsetAt(paragraphRange.start) + match.index + sentence.length,
                );
                candidates.push({
                    id: `${document.uri.fsPath}:${start.line}:${start.character}:${end.line}:${end.character}`,
                    range: new vscode.Range(start, end),
                    text: normalizeCitationScanText(sentence),
                    lexicalScore: classification.score,
                    lexicalReason: classification.reason,
                });
            }
            match = sentencePattern.exec(paragraphText);
        }
        const classification = candidates.length === 0 ? classifyCitationCandidateSentence(paragraphText) : undefined;
        if (classification) {
            candidates.push({
                id: `${document.uri.fsPath}:${paragraphRange.start.line}:${paragraphRange.start.character}:${paragraphRange.end.line}:${paragraphRange.end.character}`,
                range: paragraphRange,
                text: normalizeCitationScanText(paragraphText),
                lexicalScore: classification.score,
                lexicalReason: classification.reason,
            });
        }
        return candidates;
    }

    function shouldBreakCitationScanParagraph(line: string): boolean {
        const trimmed = line.trim();
        return (
            trimmed.length === 0 ||
            trimmed.startsWith('%') ||
            /^\\(?:section|subsection|subsubsection|paragraph|chapter|caption|label|ref|cref|begin|end|item)\b/.test(
                trimmed,
            ) ||
            /^\\(?:documentclass|usepackage|title|author|date|maketitle|addbibresource|bibliography|printbibliography)\b/.test(
                trimmed,
            )
        );
    }

    function classifyCitationCandidateSentence(sentence: string): { score: number; reason: string } | undefined {
        if (extractExistingCitations(sentence).length > 0) {
            return undefined;
        }
        const clean = normalizeCitationScanText(sentence);
        if (clean.length < 55 || clean.split(/\s+/).length < 8) {
            return undefined;
        }
        if (/^(we|our|this paper|in this paper)\b/i.test(clean) && !containsExternalClaimCue(clean)) {
            return undefined;
        }
        const cues = citationScanCueMatches(clean);
        if (cues.length === 0) {
            return undefined;
        }
        const score = Math.min(0.55 + cues.length * 0.09, 0.91);
        return {
            score,
            reason: `本地规则命中 ${cues.slice(0, 3).join(', ')}`,
        };
    }

    function containsExternalClaimCue(text: string): boolean {
        return citationScanCueMatches(text).length > 0;
    }

    function citationScanCueMatches(text: string): string[] {
        const patterns: Array<[RegExp, string]> = [
            [/\b(existing|previous|prior|recent)\b/i, '已有/近期工作'],
            [/\b(state of the art|state-of-the-art)\b/i, 'SOTA 论断'],
            [/\b(widely|generally|typically|commonly|usually|often)\b/i, '普遍性论断'],
            [/\b(many|several|various|most)\b/i, '多方法概括'],
            [/\b(traditional|conventional)\b/i, '传统方法概括'],
            [/\b(has been|have been|is known|are known|was shown|were shown|has shown|have shown)\b/i, '已有结论'],
            [/\b(reported|demonstrated|observed|found|suggested)\b/i, '实验/文献结论'],
            [/\b(survey|review|benchmark)\b/i, '综述/基准论断'],
            [/\b(outperform|outperforms|relies on|rely on|suffers from|limited by)\b/i, '比较/局限论断'],
            [/(已有|现有|以往|已有研究|已有方法)/, '已有研究'],
            [/(通常|一般|广泛|普遍)/, '普遍性论断'],
            [/(研究表明|证明|显示|报道)/, '已有结论'],
            [/(综述|基准|依赖|受限|优于)/, '比较/局限论断'],
        ];
        return patterns.filter(([pattern]) => pattern.test(text)).map(([, label]) => label);
    }

    function normalizeCitationScanText(text: string): string {
        return stripTexComment(text)
            .replace(/\\(?:cite|citet|citep|parencite|textcite|autocite)\s*(?:\[[^\]]*\]\s*){0,2}\{[^}]+\}/g, ' ')
            .replace(/\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?/g, ' $1 ')
            .replace(/[{}$]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    function stripTexComment(line: string): string {
        let escaped = false;
        for (let index = 0; index < line.length; index += 1) {
            const char = line[index];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (char === '\\') {
                escaped = true;
                continue;
            }
            if (char === '%') {
                return line.slice(0, index);
            }
        }
        return line;
    }

    context.subscriptions.push(
        vscode.languages.registerCodeActionsProvider(
            [
                { language: 'latex', scheme: 'file' },
                { language: 'tex', scheme: 'file' },
            ],
            {
                provideCodeActions(document, range, codeActionContext) {
                    const diagnostics = codeActionContext.diagnostics.filter(isCitationDiagnostic);
                    if (diagnostics.length === 0) {
                        return [];
                    }
                    const actions: vscode.CodeAction[] = [];
                    const hasAuditDiagnostic = diagnostics.some(isAuditCitationDiagnostic);
                    if (hasAuditDiagnostic) {
                        const resume = new vscode.CodeAction('EasyCite: 查看候选引用', vscode.CodeActionKind.QuickFix);
                        resume.command = {
                            title: 'EasyCite: 查看候选引用',
                            command: 'citation-agent.resumeCandidateReview',
                        };
                        actions.push(resume);
                    }

                    const reaudit = new vscode.CodeAction(
                        hasAuditDiagnostic ? 'EasyCite: 重新审计此范围' : 'EasyCite: 审计此句并推荐引用',
                        vscode.CodeActionKind.QuickFix,
                    );
                    reaudit.command = {
                        title: hasAuditDiagnostic ? 'EasyCite: 重新审计此范围' : 'EasyCite: 审计此句并推荐引用',
                        command: 'citation-agent.auditDiagnosticRange',
                        arguments: [document.uri, range],
                    };
                    actions.push(reaudit);

                    const clear = new vscode.CodeAction('EasyCite: 清除此标记', vscode.CodeActionKind.QuickFix);
                    clear.command = {
                        title: 'EasyCite: 清除此标记',
                        command: 'citation-agent.clearDiagnostic',
                        arguments: [document.uri, range],
                    };
                    actions.push(clear);
                    return actions;
                },
            },
            {
                providedCodeActionKinds: [vscode.CodeActionKind.QuickFix],
            },
        ),
    );

    function updateReviewStatus(): void {
        if (!activeReviewSession) {
            reviewStatus.hide();
            return;
        }
        const selectedCount = activeReviewSession.selectedKeys.size;
        reviewStatus.text =
            selectedCount > 0
                ? `$(references) EasyCite: 返回候选 (${selectedCount} 已选)`
                : '$(references) EasyCite: 返回候选';
        reviewStatus.tooltip = '继续查看候选证据、选择多条引用或插入引用';
        reviewStatus.show();
    }

    function endReview(): void {
        activeReviewSession = undefined;
        updateReviewStatus();
        candidatePanel?.dispose();
        candidatePanel = undefined;
    }

    async function getRunningClient(preferred?: LanguageClient): Promise<LanguageClient | undefined> {
        if (preferred?.state === State.Running) {
            return preferred;
        }

        let client = getClient();
        if (client?.state === State.Running) {
            return client;
        }

        client = await ensureServer();
        if (client?.state === State.Running) {
            return client;
        }

        return undefined;
    }

    async function openSessionDocument(session: CandidateReviewSession): Promise<vscode.TextEditor> {
        const document = await vscode.workspace.openTextDocument(session.documentUri);
        return vscode.window.showTextDocument(document, {
            preview: false,
            viewColumn: vscode.ViewColumn.One,
        });
    }

    async function exportEvalCaseFromSession(
        session: CandidateReviewSession,
        draft: EvalCaseDraft | undefined,
    ): Promise<void> {
        const state = session.result.agent_state;
        const runId = state?.run_id ?? session.result.trace_events?.[0]?.run_id ?? Date.now().toString(36);
        const defaultId = `badcase_${safeEvalId(runId).slice(0, 16)}`;
        const id = safeEvalId(draft?.id || defaultId);
        const notes = String(draft?.notes ?? `Exported from EasyCite run ${runId}.`).trim();

        const request = state?.request;
        const evalCase = {
            id,
            claim: request?.claim || session.claim,
            context: request?.context || session.claim,
            citation_needed: session.result.citation_needed,
            gold_papers: splitEvalList(draft?.gold),
            acceptable_papers: splitEvalList(draft?.acceptable),
            forbidden_papers: splitEvalList(draft?.forbidden),
            expected_evidence_keywords: [],
            notes,
        };
        const workspacePath = getWorkspacePath(await vscode.workspace.openTextDocument(session.documentUri));
        const evalDir = path.join(
            workspacePath || path.dirname(session.documentUri.fsPath),
            '.citation-agent',
            'eval_cases',
        );
        await fs.promises.mkdir(evalDir, { recursive: true });
        const evalPath = path.join(evalDir, 'user_bad_cases.jsonl');
        await fs.promises.appendFile(evalPath, `${JSON.stringify(evalCase)}\n`, 'utf-8');
        traceInfo(`EasyCite exported eval bad case: ${evalPath}#${evalCase.id}`);
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(evalPath));
        await vscode.window.showTextDocument(document, {
            preview: false,
            viewColumn: vscode.ViewColumn.Beside,
        });
        vscode.window.showInformationMessage(`EasyCite: 已导出 eval bad case 到 ${evalPath}`);
    }

    function splitEvalList(value: string | undefined): string[] {
        return (value ?? '')
            .split(/[;；\n]/)
            .map((item) => item.trim())
            .filter(Boolean);
    }

    function safeEvalId(value: string): string {
        const normalized = value
            .trim()
            .replace(/[^a-zA-Z0-9_-]+/g, '_')
            .replace(/^_+|_+$/g, '');
        return normalized || `case_${Date.now().toString(36)}`;
    }

    function refreshCandidatePanel(): void {
        if (!candidatePanel || !activeReviewSession) {
            return;
        }
        candidatePanel.webview.html = renderCandidatePanelHtml(candidatePanel.webview, activeReviewSession);
    }

    function getCandidateByPanelIndex(
        session: CandidateReviewSession,
        index: number | undefined,
    ): CitationSuggestion | undefined {
        if (typeof index !== 'number') {
            return undefined;
        }
        const candidates = [...session.result.suggestions, ...(session.result.diagnostic_suggestions ?? [])];
        return candidates[index];
    }

    async function recordCandidateFeedback(
        session: CandidateReviewSession,
        suggestion: CitationSuggestion,
        action: 'accepted' | 'rejected',
    ): Promise<void> {
        const config = vscode.workspace.getConfiguration('citation-agent');
        if (!config.get<boolean>('memoryEnabled', true)) {
            return;
        }
        try {
            const document = await vscode.workspace.openTextDocument(session.documentUri);
            await withTimeout(
                session.client.sendRequest(RecordFeedbackRequest, {
                    workspace_path: getWorkspacePath(document),
                    file_path: document.uri.fsPath,
                    claim: session.claim,
                    context: getParagraphContext(document, session.selection),
                    action,
                    suggestion,
                }),
                5_000,
            );
            traceInfo(`EasyCite memory feedback recorded: ${action} ${suggestion.title}`);
        } catch (error) {
            traceError(`EasyCite memory feedback failed: ${error}`);
        }
    }

    async function recordAcceptedCitationKeys(session: CandidateReviewSession, citationKeys: string[]): Promise<void> {
        const suggestions = suggestionsForCitationKeys(allSuggestions(session.result), citationKeys);
        await Promise.all(suggestions.map((suggestion) => recordCandidateFeedback(session, suggestion, 'accepted')));
    }

    async function insertSessionCitation(session: CandidateReviewSession, citationKeys: string[]): Promise<void> {
        if (citationKeys.length === 0) {
            vscode.window.showWarningMessage('EasyCite: 请先选择至少一条有 citation key 的候选引用。');
            return;
        }
        const liveClient = await getRunningClient(session.client);
        if (!liveClient) {
            vscode.window.showErrorMessage('EasyCite: Python 后端尚未启动，无法补全 BibTeX。请检查输出日志。');
            return;
        }
        session.client = liveClient;
        const editor = await openSessionDocument(session);
        const applied = await confirmAndApplyCitationPlan(
            liveClient,
            editor,
            session.selection,
            citationKeys,
            allSuggestions(session.result),
        );
        if (applied) {
            await recordAcceptedCitationKeys(session, citationKeys);
            endReview();
        }
    }

    function openCandidatePanel(session: CandidateReviewSession): void {
        if (candidatePanel) {
            candidatePanel.reveal(vscode.ViewColumn.Beside);
            refreshCandidatePanel();
            return;
        }

        candidatePanel = vscode.window.createWebviewPanel(
            'citationAgentCandidates',
            'EasyCite Candidates',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            },
        );
        candidatePanel.webview.html = renderCandidatePanelHtml(candidatePanel.webview, session);
        candidatePanel.onDidDispose(
            () => {
                candidatePanel = undefined;
                updateReviewStatus();
            },
            undefined,
            context.subscriptions,
        );
        candidatePanel.webview.onDidReceiveMessage(
            async (message: CandidatePanelMessage) => {
                if (!activeReviewSession) {
                    return;
                }

                const activeSession = activeReviewSession;
                const suggestion = getCandidateByPanelIndex(activeSession, message.index);

                if (message.command === 'showEvidence' && suggestion) {
                    activeSession.expandedEvidenceIndex =
                        activeSession.expandedEvidenceIndex === message.index ? undefined : message.index;
                    refreshCandidatePanel();
                    return;
                }

                if (message.command === 'toggle' && suggestion?.citation_key) {
                    if (activeSession.selectedKeys.has(suggestion.citation_key)) {
                        activeSession.selectedKeys.delete(suggestion.citation_key);
                        await recordCandidateFeedback(activeSession, suggestion, 'rejected');
                    } else {
                        activeSession.selectedKeys.add(suggestion.citation_key);
                        await recordCandidateFeedback(activeSession, suggestion, 'accepted');
                    }
                    updateReviewStatus();
                    refreshCandidatePanel();
                    return;
                }

                if (message.command === 'insertOne' && suggestion?.citation_key) {
                    await insertSessionCitation(activeSession, [suggestion.citation_key]);
                    return;
                }

                if (message.command === 'insertSelected') {
                    await insertSessionCitation(activeSession, [...activeSession.selectedKeys]);
                    return;
                }

                if (message.command === 'clearSelected') {
                    activeSession.selectedKeys.clear();
                    updateReviewStatus();
                    refreshCandidatePanel();
                    return;
                }

                if (message.command === 'endReview') {
                    endReview();
                }
            },
            undefined,
            context.subscriptions,
        );
    }

    function getActiveReviewSessionForTool(): CandidateReviewSession | undefined {
        if (!activeReviewSession) {
            vscode.window.showInformationMessage('EasyCite: 当前没有可查看的候选审核。请先运行“检查选中内容”。');
            return undefined;
        }
        return activeReviewSession;
    }

    async function openRunRecordForSession(session: CandidateReviewSession): Promise<void> {
        const runPath = session.result.agent_run_path;
        if (!runPath) {
            vscode.window.showWarningMessage('EasyCite: 当前运行没有保存 run record。');
            return;
        }
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(runPath));
        await vscode.window.showTextDocument(document, {
            preview: false,
            viewColumn: vscode.ViewColumn.Beside,
        });
    }

    function openAgentStateTool(session: CandidateReviewSession): void {
        const body = [
            renderAgentStateSection(session.result, false, false),
            renderTargetTraceSection(session.result.target_trace ?? []),
        ]
            .filter(Boolean)
            .join('');
        const panel = vscode.window.createWebviewPanel(
            'citationAgentState',
            'EasyCite State',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            },
        );
        panel.webview.html = renderStandaloneToolHtml(
            panel.webview,
            'EasyCite State',
            body || '<p class="muted">当前运行没有 Agent State。</p>',
            `
            const vscode = acquireVsCodeApi();
            document.addEventListener('click', (event) => {
                const button = event.target.closest('button[data-command]');
                if (button?.getAttribute('data-command') === 'openRunRecord') {
                    vscode.postMessage({ command: 'openRunRecord' });
                }
            });
            `,
        );
        panel.webview.onDidReceiveMessage(
            async (message: DiagnosticPanelMessage) => {
                if (message.command === 'openRunRecord') {
                    await openRunRecordForSession(session);
                }
            },
            undefined,
            context.subscriptions,
        );
    }

    function openRunTraceTool(session: CandidateReviewSession): void {
        const body = [
            renderRunTraceSection(session.result.trace_events ?? []),
            renderTargetTraceSection(session.result.target_trace ?? []),
        ]
            .filter(Boolean)
            .join('');
        const panel = vscode.window.createWebviewPanel(
            'citationAgentRunTrace',
            'EasyCite Run Trace',
            vscode.ViewColumn.Beside,
            {
                enableScripts: false,
                retainContextWhenHidden: true,
            },
        );
        panel.webview.html = renderStandaloneToolHtml(
            panel.webview,
            'EasyCite Run Trace',
            body || '<p class="muted">当前运行没有 Trace。</p>',
        );
    }

    function openEvalBadCaseTool(session: CandidateReviewSession): void {
        if (!vscode.workspace.getConfiguration('citation-agent').get<boolean>('developerMode', false)) {
            vscode.window.showWarningMessage('EasyCite: Eval Bad Case 导出仅在 developerMode 下可用。');
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            'citationAgentEvalBadCase',
            'EasyCite Eval Bad Case',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            },
        );
        panel.webview.html = renderEvalCaseToolHtml(panel.webview, session);
        panel.webview.onDidReceiveMessage(
            async (message: DiagnosticPanelMessage) => {
                if (message.command !== 'saveEvalCase') {
                    return;
                }
                await exportEvalCaseFromSession(session, message.evalCase);
                panel.dispose();
            },
            undefined,
            context.subscriptions,
        );
    }

    context.subscriptions.push(
        vscode.commands.registerCommand('citation-agent.resumeCandidateReview', async () => {
            if (!activeReviewSession) {
                vscode.window.showInformationMessage('EasyCite: 当前没有可恢复的候选审核。');
                return;
            }

            openCandidatePanel(activeReviewSession);
        }),
        vscode.commands.registerCommand('citation-agent.openAgentState', async () => {
            const session = getActiveReviewSessionForTool();
            if (session) {
                openAgentStateTool(session);
            }
        }),
        vscode.commands.registerCommand('citation-agent.openRunTrace', async () => {
            const session = getActiveReviewSessionForTool();
            if (session) {
                openRunTraceTool(session);
            }
        }),
        vscode.commands.registerCommand('citation-agent.exportEvalBadCase', async () => {
            const session = getActiveReviewSessionForTool();
            if (session) {
                openEvalBadCaseTool(session);
            }
        }),
        vscode.commands.registerCommand('citation-agent.clearProjectMemory', async () => {
            const liveClient = await getRunningClient(activeReviewSession?.client);
            if (!liveClient) {
                vscode.window.showErrorMessage('EasyCite: Python 后端尚未启动，无法清空项目记忆。');
                return;
            }
            const workspacePath = activeReviewSession
                ? getWorkspacePath(await vscode.workspace.openTextDocument(activeReviewSession.documentUri))
                : (vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '');
            if (!workspacePath) {
                vscode.window.showWarningMessage('EasyCite: 当前没有可清空记忆的工作区。');
                return;
            }
            const confirmed = await vscode.window.showWarningMessage(
                `EasyCite: 确认清空当前项目的反馈记忆？\n${workspacePath}`,
                { modal: true },
                '清空',
            );
            if (confirmed !== '清空') {
                return;
            }
            const result = await withTimeout(
                liveClient.sendRequest(ClearMemoryRequest, { workspace_path: workspacePath }),
                10_000,
            );
            if (result.error) {
                vscode.window.showErrorMessage(`EasyCite: 清空项目记忆失败：${result.error}`);
                return;
            }
            vscode.window.showInformationMessage(`EasyCite: 已清空项目记忆 ${result.count ?? 0} 条。`);
        }),
        vscode.commands.registerCommand('citation-agent.clearDiagnostic', (uri?: vscode.Uri, range?: vscode.Range) => {
            clearCitationDiagnostics(uri, range);
        }),
        vscode.commands.registerCommand('citation-agent.clearAllDiagnostics', () => {
            clearCitationDiagnostics();
        }),
        vscode.commands.registerCommand('citation-agent.scanTexProject', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !isTexDocument(editor.document)) {
                vscode.window.showWarningMessage('EasyCite: 请先打开一个 TeX/LaTeX 文档。');
                return;
            }
            await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: 'EasyCite: 正在扫描 TeX 工程引用标记',
                    cancellable: false,
                },
                async () => scanTexProjectForCitationDiagnostics(editor.document, 'manual-command'),
            );
            vscode.window.setStatusBarMessage('EasyCite: TeX 工程引用标记扫描完成。', 5000);
        }),
        vscode.commands.registerCommand(
            'citation-agent.auditDiagnosticRange',
            async (uri?: vscode.Uri, range?: vscode.Range) => {
                if (!uri || !range) {
                    vscode.window.showWarningMessage('EasyCite: 没有可重新审计的诊断范围。');
                    return;
                }
                const document = await vscode.workspace.openTextDocument(uri);
                const editor = await vscode.window.showTextDocument(document, {
                    preview: false,
                    viewColumn: vscode.ViewColumn.One,
                });
                editor.selection = new vscode.Selection(range.start, range.end);
                await vscode.commands.executeCommand('citation-agent.auditSelection', { forceRefresh: true });
            },
        ),
    );

    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument((document) => scheduleTexProjectScan(document, 'open-document')),
        vscode.workspace.onDidSaveTextDocument((document) => scheduleTexProjectScan(document, 'save-document')),
        vscode.workspace.onDidChangeTextDocument((event) => scheduleTexProjectScan(event.document, 'change-document')),
        vscode.window.onDidChangeActiveTextEditor((editor) =>
            scheduleTexProjectScan(editor?.document, 'active-editor'),
        ),
    );

    for (const document of vscode.workspace.textDocuments) {
        scheduleTexProjectScan(document, 'activation-open-document');
    }
    scheduleTexProjectScan(vscode.window.activeTextEditor?.document, 'activation-active-editor');

    context.subscriptions.push(
        vscode.commands.registerCommand(
            'citation-agent.auditSelection',
            async (options?: { forceRefresh?: boolean }) => {
                if (auditInFlight) {
                    vscode.window.showInformationMessage('EasyCite: 正在检查上一段内容，请等待当前请求完成。');
                    return;
                }

                const editor = vscode.window.activeTextEditor;
                if (!editor) {
                    vscode.window.showWarningMessage('EasyCite: 请先打开一个 LaTeX 文档。');
                    return;
                }

                const selection = editor.selection;
                if (selection.isEmpty) {
                    vscode.window.showWarningMessage('EasyCite: 请先选中需要检查的句子或文本。');
                    return;
                }

                const forceRefresh = Boolean(options?.forceRefresh);
                const params = buildAuditParams(editor.document, selection, forceRefresh);
                if (!params.claim) {
                    vscode.window.showWarningMessage('EasyCite: 选中文本为空。');
                    return;
                }

                let client = getClient();
                if (!client || client.state !== State.Running) {
                    client = await ensureServer();
                }

                if (!client || client.state !== State.Running) {
                    vscode.window.showErrorMessage('EasyCite: Python 后端尚未启动，请检查输出日志。');
                    return;
                }

                const zoteroAvailable = await ensureZoteroAvailable();
                if (!zoteroAvailable) {
                    return;
                }

                traceInfo(`EasyCite audit request: ${JSON.stringify(params, null, 2)}`);
                auditInFlight = true;
                try {
                    const result = await vscode.window.withProgress(
                        {
                            location: vscode.ProgressLocation.Notification,
                            title: 'EasyCite: 正在检查引用候选',
                            cancellable: false,
                        },
                        async () =>
                            withTimeout(client.sendRequest(AuditClaimRequest, params), AUDIT_REQUEST_TIMEOUT_MS),
                    );
                    traceLog(`EasyCite audit result: ${JSON.stringify(result, null, 2)}`);
                    auditInFlight = false;
                    const hasCandidates =
                        result.suggestions.length > 0 || (result.diagnostic_suggestions?.length ?? 0) > 0;
                    if (result.citation_needed) {
                        upsertCitationDiagnostic(
                            editor.document.uri,
                            selection,
                            result,
                            result.suggestions.length + (result.diagnostic_suggestions?.length ?? 0),
                        );
                    } else {
                        clearCitationDiagnostics(editor.document.uri, selection);
                    }
                    if (result.citation_needed || hasCandidates) {
                        const selectedKeys = new Set<string>();
                        activeReviewSession = {
                            documentUri: editor.document.uri,
                            selection,
                            claim: params.claim,
                            result,
                            client,
                            selectedKeys,
                        };
                        updateReviewStatus();
                        openCandidatePanel(activeReviewSession);
                        vscode.window.setStatusBarMessage(renderSummary(result), 8000);
                    } else {
                        vscode.window.showInformationMessage(renderSummary(result));
                    }
                } catch (error) {
                    traceError(`EasyCite audit failed: ${error}`);
                    vscode.window.showErrorMessage(`EasyCite: 检查失败，详情见输出日志。`);
                } finally {
                    auditInFlight = false;
                }
            },
        ),
    );
}
