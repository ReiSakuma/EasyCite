import { ParameterStructures, RequestType } from 'vscode-languageclient';

export const AUDIT_CLAIM_METHOD = 'citationAgent/auditClaim';
export const TRANSLATE_EVIDENCE_METHOD = 'citationAgent/translateEvidence';
export const RESOLVE_BIBTEX_METHOD = 'citationAgent/resolveBibtex';
export const RANK_CITATION_SCAN_METHOD = 'citationAgent/rankCitationScan';
export const RECORD_FEEDBACK_METHOD = 'citationAgent/recordFeedback';
export const CLEAR_MEMORY_METHOD = 'citationAgent/clearMemory';
export const TOOL_HEALTH_METHOD = 'citationAgent/toolHealth';

export interface AuditClaimParams {
    workspace_path: string;
    file_path: string;
    line_start: number;
    line_end: number;
    claim: string;
    context: string;
    existing_citations: string[];
    force_refresh?: boolean;
}

export interface EvidenceSnippet {
    page: number | null;
    section: string | null;
    text: string | null;
    source: 'metadata' | 'abstract' | 'pdf' | 'unknown';
    score: number | null;
    quality_score?: number | null;
    is_own_claim?: boolean | null;
    is_related_work_summary?: boolean | null;
    supports_core_mechanism?: boolean | null;
    citation_density?: number | null;
    quality_flags?: string[];
}

export interface TraceEvent {
    run_id: string;
    stage: string;
    message: string;
    candidate_count_before: number | null;
    candidate_count_after: number | null;
    latency_ms: number | null;
    error: string | null;
    related_candidate_keys: string[];
}

export interface AgentRunState {
    schema_version?: string;
    run_id?: string;
    created_at?: string;
    request?: {
        workspace_path?: string;
        file_path?: string;
        line_start?: number | null;
        line_end?: number | null;
        claim?: string;
        context?: string;
        existing_citations?: string[];
    };
    runtime?: {
        model?: Record<string, unknown>;
        sources?: Record<string, unknown>;
    };
    claim_profile?: Record<string, unknown>;
    retrieval_plan?: Record<string, unknown>;
    context_engineering?: {
        schema_version?: string;
        reports?: Array<Record<string, unknown>>;
        total_calls?: number;
        estimated_tokens_before?: number;
        estimated_tokens_after?: number;
    };
    tool_reliability?: {
        schema_version?: string;
        observations?: Array<Record<string, unknown>>;
        total_calls?: number;
        successful_calls?: number;
        failed_calls?: number;
        fallback_calls?: number;
        error_categories?: string[];
    };
    candidate_pool?: {
        counts?: Record<string, number>;
    };
    output?: {
        recommendation_count?: number;
        diagnostic_count?: number;
        citation_keys?: string[];
    };
    graph?: {
        schema_version?: string;
        framework?: string;
        runtime?: string;
        nodes?: Array<Record<string, unknown>>;
        edges?: Array<Record<string, unknown>>;
        checkpoint_policy?: Record<string, unknown>;
    };
}

export interface CitationSuggestion {
    paper_id: string;
    title: string;
    citation_key: string | null;
    authors: string[];
    year: string | null;
    venue: string | null;
    doi: string | null;
    url: string | null;
    pdf_path: string | null;
    verdict:
        | 'metadata_match'
        | 'foundational_support'
        | 'direct_support'
        | 'partial_support'
        | 'related_only'
        | 'contradiction'
        | 'insufficient_evidence';
    confidence: number;
    source: 'mock' | 'current_bib' | 'zotero' | 'online';
    evidence: EvidenceSnippet;
    evidence_passages: EvidenceSnippet[];
    trace?: string[];
    reason: string;
}

export interface AuditClaimResult {
    citation_needed: boolean;
    reason: string;
    confidence: number;
    claim_type:
        | 'background'
        | 'related_work_summary'
        | 'method_claim'
        | 'empirical_result'
        | 'definition'
        | 'common_knowledge'
        | 'writing_only'
        | 'other';
    source: 'llm' | 'heuristic';
    suggestions: CitationSuggestion[];
    diagnostic_suggestions?: CitationSuggestion[];
    target_trace?: string[];
    trace_events?: TraceEvent[];
    agent_state?: AgentRunState;
    agent_run_path?: string | null;
    tool_warnings?: string[];
    recommended_rewrite: string | null;
}

export interface TranslateEvidenceText {
    id: string;
    text: string;
}

export interface TranslateEvidenceParams {
    texts: TranslateEvidenceText[];
}

export interface EvidenceTranslation {
    id: string;
    zh: string;
}

export interface TranslateEvidenceResult {
    available: boolean;
    reason: string;
    translations: EvidenceTranslation[];
}

export interface ResolveBibtexParams {
    citation_keys: string[];
    suggestions: CitationSuggestion[];
}

export interface ResolvedBibtexEntry {
    citation_key: string;
    bibtex: string;
    source: string;
    source_trust?: 'high' | 'medium' | 'low' | string;
    source_rank?: number;
    requires_user_verification?: boolean;
    quality_warnings?: string[];
}

export interface ResolveBibtexMissingEntry {
    citation_key: string;
    reason: string;
}

export interface ResolveBibtexResult {
    entries: ResolvedBibtexEntry[];
    missing: ResolveBibtexMissingEntry[];
}

export interface CitationScanSentence {
    id: string;
    text: string;
}

export interface RankCitationScanParams {
    sentences: CitationScanSentence[];
}

export interface CitationScanScore {
    id: string;
    score: number;
    matched_prototype: string;
}

export interface RankCitationScanResult {
    available: boolean;
    reason: string;
    provider: string | null;
    model_id: string | null;
    scores: CitationScanScore[];
}

export interface RecordFeedbackParams {
    workspace_path: string;
    file_path: string;
    claim: string;
    context: string;
    action: 'accepted' | 'rejected';
    suggestion: CitationSuggestion;
}

export interface RecordFeedbackResult {
    recorded?: boolean;
    cleared?: boolean;
    count?: number;
    path?: string;
    error?: string;
}

export type ToolHealthStatus = 'ok' | 'warning' | 'error' | 'disabled';

export interface ToolHealthItem {
    name: string;
    label: string;
    status: ToolHealthStatus;
    summary: string;
    details: string[];
    actions: string[];
    latency_ms: number | null;
}

export interface ToolHealthResult {
    schema_version: string;
    workspace_path: string;
    file_path: string;
    generated_at: string;
    summary: {
        ok: number;
        warning: number;
        error: number;
        disabled: number;
        overall_status: ToolHealthStatus;
        message: string;
    };
    items: ToolHealthItem[];
}

export const AuditClaimRequest = new RequestType<AuditClaimParams, AuditClaimResult, void>(
    AUDIT_CLAIM_METHOD,
    ParameterStructures.byName,
);

export const TranslateEvidenceRequest = new RequestType<TranslateEvidenceParams, TranslateEvidenceResult, void>(
    TRANSLATE_EVIDENCE_METHOD,
    ParameterStructures.byName,
);

export const ResolveBibtexRequest = new RequestType<ResolveBibtexParams, ResolveBibtexResult, void>(
    RESOLVE_BIBTEX_METHOD,
    ParameterStructures.byName,
);

export const RankCitationScanRequest = new RequestType<RankCitationScanParams, RankCitationScanResult, void>(
    RANK_CITATION_SCAN_METHOD,
    ParameterStructures.byName,
);

export const RecordFeedbackRequest = new RequestType<RecordFeedbackParams, RecordFeedbackResult, void>(
    RECORD_FEEDBACK_METHOD,
    ParameterStructures.byName,
);

export const ClearMemoryRequest = new RequestType<{ workspace_path: string }, RecordFeedbackResult, void>(
    CLEAR_MEMORY_METHOD,
    ParameterStructures.byName,
);

export const ToolHealthRequest = new RequestType<{ workspace_path: string; file_path: string }, ToolHealthResult, void>(
    TOOL_HEALTH_METHOD,
    ParameterStructures.byName,
);
