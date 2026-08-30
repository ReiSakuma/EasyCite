# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""EasyCite v2 language server.

V2 detects whether a selected LaTeX claim needs a citation, then searches
available local citation sources for candidate papers.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from collections.abc import Mapping, Sequence
from typing import Any, Optional


def update_sys_path(path_to_add: str, strategy: str) -> None:
    """Add bundled libraries to sys.path."""
    if path_to_add not in sys.path and os.path.isdir(path_to_add):
        if strategy == "useBundled":
            sys.path.insert(0, path_to_add)
        elif strategy == "fromEnvironment":
            sys.path.append(path_to_add)


update_sys_path(
    os.fspath(pathlib.Path(__file__).parent.parent / "libs"),
    os.getenv("LS_IMPORT_STRATEGY", "useBundled"),
)
update_sys_path(
    os.getenv("CITATION_AGENT_PRIVATE_LIBS", ""),
    "fromEnvironment",
)

# pylint: disable=wrong-import-position,import-error
import lsp_jsonrpc as jsonrpc
from citation_agent.bibtex_resolver import resolve_bibtex_entries
from citation_agent.citation_scan import rank_citation_scan_sentences
from citation_agent.detection import CitationNeedDetector
from citation_agent.memory import clear_candidate_feedback, record_candidate_feedback
from citation_agent.recommender import CitationRecommender
from citation_agent.tool_health import run_tool_health_check
from citation_agent.tools.observability import get_tool_observations, reset_tool_observations
from citation_agent.translation import translate_evidence
from lsprotocol import types as lsp
from pygls import uris, workspace
from pygls.lsp.server import LanguageServer

WORKSPACE_SETTINGS: dict[str, dict[str, Any]] = {}
GLOBAL_SETTINGS: dict[str, Any] = {}
AUDIT_CLAIM_METHOD = "citationAgent/auditClaim"
TRANSLATE_EVIDENCE_METHOD = "citationAgent/translateEvidence"
RESOLVE_BIBTEX_METHOD = "citationAgent/resolveBibtex"
RANK_CITATION_SCAN_METHOD = "citationAgent/rankCitationScan"
RECORD_FEEDBACK_METHOD = "citationAgent/recordFeedback"
CLEAR_MEMORY_METHOD = "citationAgent/clearMemory"
TOOL_HEALTH_METHOD = "citationAgent/toolHealth"

MAX_WORKERS = 5
LSP_SERVER = LanguageServer(
    name="EasyCite",
    version="0.2.0",
    max_workers=MAX_WORKERS,
)


AUDIT_PARAM_FIELDS = (
    "workspace_path",
    "file_path",
    "line_start",
    "line_end",
    "claim",
    "context",
    "existing_citations",
    "force_refresh",
)


def normalize_audit_params(params: Any) -> dict[str, Any]:
    """Normalize pygls custom request params to a plain dict.

    JSON-RPC supports both object and positional params. The VS Code client now
    sends named params, but this keeps v1 tolerant of already-running clients or
    older request shapes.
    """
    if isinstance(params, Mapping):
        return dict(params)

    if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        return {
            field: params[index] if index < len(params) else None
            for index, field in enumerate(AUDIT_PARAM_FIELDS)
        }

    raise TypeError(f"Unsupported audit_claim params type: {type(params).__name__}")


def normalize_translate_params(params: Any) -> dict[str, Any]:
    """Normalize custom translateEvidence request params to a plain dict."""
    if isinstance(params, Mapping):
        return dict(params)

    if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        if params and isinstance(params[0], Mapping):
            first = dict(params[0])
            if "texts" in first:
                return first
            return {"texts": list(params)}
        if params and isinstance(params[0], list):
            return {"texts": params[0]}
        return {"texts": list(params)}

    return {}


def normalize_resolve_bibtex_params(params: Any) -> dict[str, Any]:
    """Normalize custom resolveBibtex request params to a plain dict."""
    if isinstance(params, Mapping):
        return dict(params)

    if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        if params and isinstance(params[0], Mapping):
            first = dict(params[0])
            if "citation_keys" in first or "suggestions" in first:
                return first
        citation_keys = params[0] if len(params) > 0 and isinstance(params[0], list) else []
        suggestions = params[1] if len(params) > 1 and isinstance(params[1], list) else []
        return {
            "citation_keys": citation_keys,
            "suggestions": suggestions,
        }

    return {}


def normalize_rank_citation_scan_params(params: Any) -> dict[str, Any]:
    """Normalize custom rankCitationScan request params to a plain dict."""
    if isinstance(params, Mapping):
        return dict(params)

    if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        if params and isinstance(params[0], Mapping):
            first = dict(params[0])
            if "sentences" in first:
                return first
        if params and isinstance(params[0], list):
            return {"sentences": params[0]}
        return {"sentences": list(params)}

    return {}


def normalize_record_feedback_params(params: Any) -> dict[str, Any]:
    """Normalize custom feedback-memory request params to a plain dict."""
    if isinstance(params, Mapping):
        return dict(params)

    if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        if params and isinstance(params[0], Mapping):
            return dict(params[0])
        return {}

    return {}


def normalize_tool_health_params(params: Any) -> dict[str, Any]:
    """Normalize custom toolHealth request params to a plain dict."""
    if isinstance(params, Mapping):
        return dict(params)

    if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        if params and isinstance(params[0], Mapping):
            return dict(params[0])
        return {}

    return {}


def audit_claim(params: Any) -> dict[str, Any]:
    """Return citation-need detection plus local candidate recommendations."""
    params = normalize_audit_params(params)
    reset_tool_observations()
    detector = CitationNeedDetector()
    result = detector.detect(params)
    if result.citation_needed:
        recommendation = CitationRecommender(logger=log_to_output).recommend_with_diagnostics(
            {**params, "_preserve_tool_observations": True}
        )
        result = result.model_copy(
            update={
                "suggestions": recommendation.suggestions,
                "diagnostic_suggestions": recommendation.diagnostic_suggestions,
                "target_trace": recommendation.target_trace,
                "trace_events": recommendation.trace_events,
                "agent_state": recommendation.agent_state,
                "agent_run_path": recommendation.agent_run_path,
                "tool_warnings": recommendation.tool_warnings,
            }
        )
    return result.model_dump()


@LSP_SERVER.feature(AUDIT_CLAIM_METHOD)
def audit_claim_request(_ls: LanguageServer, params: Any) -> dict[str, Any]:
    """LSP custom request handler for EasyCite v1."""
    log_to_output(
        f"Received {AUDIT_CLAIM_METHOD}: "
        f"{json.dumps(params, ensure_ascii=False)}"
    )
    return run_handler_safely("auditClaim", lambda: audit_claim(params))


@LSP_SERVER.feature(TRANSLATE_EVIDENCE_METHOD)
def translate_evidence_request(_ls: LanguageServer, params: Any) -> dict[str, Any]:
    """Translate evidence page text for bilingual review."""
    normalized = normalize_translate_params(params)
    log_to_output(
        f"Received {TRANSLATE_EVIDENCE_METHOD}: "
        f"{json.dumps({'param_type': type(params).__name__, 'text_count': len(normalized.get('texts') or [])}, ensure_ascii=False)}"
    )
    reset_tool_observations()
    result = run_handler_safely(
        "translateEvidence",
        lambda: translate_evidence(normalized, logger=log_to_output),
    )
    result["tool_observations"] = get_tool_observations()
    return result


@LSP_SERVER.feature(RESOLVE_BIBTEX_METHOD)
def resolve_bibtex_request(_ls: LanguageServer, params: Any) -> dict[str, Any]:
    """Resolve missing BibTeX entries for accepted citation suggestions."""
    normalized = normalize_resolve_bibtex_params(params)
    log_to_output(
        f"Received {RESOLVE_BIBTEX_METHOD}: "
        f"{json.dumps({'citation_key_count': len(normalized.get('citation_keys') or [])}, ensure_ascii=False)}"
    )
    reset_tool_observations()
    result = run_handler_safely(
        "resolveBibtex",
        lambda: resolve_bibtex_entries(normalized, logger=log_to_output),
    )
    result["tool_observations"] = get_tool_observations()
    return result


@LSP_SERVER.feature(RANK_CITATION_SCAN_METHOD)
def rank_citation_scan_request(_ls: LanguageServer, params: Any) -> dict[str, Any]:
    """Rank candidate sentences for background citation-needed diagnostics."""
    normalized = normalize_rank_citation_scan_params(params)
    log_to_output(
        f"Received {RANK_CITATION_SCAN_METHOD}: "
        f"{json.dumps({'sentence_count': len(normalized.get('sentences') or [])}, ensure_ascii=False)}"
    )
    reset_tool_observations()
    result = run_handler_safely(
        "rankCitationScan",
        lambda: rank_citation_scan_sentences(normalized, logger=log_to_output),
    )
    result["tool_observations"] = get_tool_observations()
    return result


@LSP_SERVER.feature(RECORD_FEEDBACK_METHOD)
def record_feedback_request(_ls: LanguageServer, params: Any) -> dict[str, Any]:
    """Persist accepted/rejected candidate feedback for project-local memory."""
    normalized = normalize_record_feedback_params(params)
    log_to_output(
        f"Received {RECORD_FEEDBACK_METHOD}: "
        f"{json.dumps({'action': normalized.get('action'), 'title': (normalized.get('suggestion') or {}).get('title') if isinstance(normalized.get('suggestion'), Mapping) else None}, ensure_ascii=False)}"
    )
    return run_handler_safely(
        "recordFeedback",
        lambda: record_candidate_feedback(normalized),
    )


@LSP_SERVER.feature(CLEAR_MEMORY_METHOD)
def clear_memory_request(_ls: LanguageServer, params: Any) -> dict[str, Any]:
    """Clear project-local candidate feedback memory."""
    normalized = normalize_record_feedback_params(params)
    log_to_output(f"Received {CLEAR_MEMORY_METHOD}")
    return run_handler_safely(
        "clearMemory",
        lambda: clear_candidate_feedback(normalized),
    )


@LSP_SERVER.feature(TOOL_HEALTH_METHOD)
def tool_health_request(_ls: LanguageServer, params: Any) -> dict[str, Any]:
    """Run read-only health checks for EasyCite tools."""
    normalized = normalize_tool_health_params(params)
    log_to_output(
        f"Received {TOOL_HEALTH_METHOD}: "
        f"{json.dumps({'workspace_path': normalized.get('workspace_path'), 'file_path': normalized.get('file_path')}, ensure_ascii=False)}"
    )
    return run_handler_safely(
        "toolHealth",
        lambda: run_tool_health_check(normalized, logger=log_to_output),
    )


def run_handler_safely(name: str, callback):
    """Run a request handler without letting library prints corrupt LSP stdio."""
    stdout = StringIO()
    stderr = StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = callback()
    except Exception:  # pylint: disable=broad-except
        captured = format_captured_stdio(stdout.getvalue(), stderr.getvalue())
        if captured:
            log_to_output(f"Captured stdio before {name} failure:\r\n{captured}")
        log_error(f"EasyCite {name} failed:\r\n{traceback.format_exc()}")
        raise

    captured = format_captured_stdio(stdout.getvalue(), stderr.getvalue())
    if captured:
        log_to_output(f"Captured stdio during {name}:\r\n{captured}")
    return result


def format_captured_stdio(stdout: str, stderr: str, limit: int = 4000) -> str:
    parts: list[str] = []
    if stdout.strip():
        parts.append(f"[stdout]\r\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"[stderr]\r\n{stderr.strip()}")
    captured = "\r\n".join(parts)
    if len(captured) > limit:
        return captured[:limit] + "\r\n... <truncated>"
    return captured


@LSP_SERVER.feature(lsp.INITIALIZE)
def initialize(_ls: LanguageServer, params: lsp.InitializeParams) -> None:
    """LSP handler for initialize request."""
    log_to_output(f"EasyCite server cwd: {os.getcwd()}")

    paths = "\r\n   ".join(sys.path)
    log_to_output(f"sys.path used to run EasyCite:\r\n   {paths}")

    initialization_options = params.initialization_options or {}
    GLOBAL_SETTINGS.update(**initialization_options.get("globalSettings", {}))

    settings = initialization_options.get("settings", [])
    _update_workspace_settings(settings)
    log_to_output(
        "Settings used to run EasyCite:\r\n"
        f"{json.dumps(settings, indent=4, ensure_ascii=False)}\r\n"
    )
    log_to_output(
        "Global settings:\r\n"
        f"{json.dumps(GLOBAL_SETTINGS, indent=4, ensure_ascii=False)}\r\n"
    )


@LSP_SERVER.feature(lsp.EXIT)
def on_exit(_ls: LanguageServer, _params: Optional[Any] = None) -> None:
    """Handle clean up on exit."""
    jsonrpc.shutdown_json_rpc()


@LSP_SERVER.feature(lsp.SHUTDOWN)
def on_shutdown(_ls: LanguageServer, _params: Optional[Any] = None) -> None:
    """Handle clean up on shutdown."""
    jsonrpc.shutdown_json_rpc()


def get_cwd(settings: dict[str, Any], document: Optional[workspace.TextDocument]) -> str:
    """Return the configured working directory with VS Code variables resolved."""
    cwd = settings.get("cwd", settings["workspaceFS"])
    workspace_fs = settings["workspaceFS"]

    if document and document.path:
        file_path = document.path
        file_dir = os.path.dirname(file_path)
        file_basename = os.path.basename(file_path)
        file_stem, file_ext = os.path.splitext(file_basename)

        substitutions = {
            "${file}": file_path,
            "${fileBasename}": file_basename,
            "${fileBasenameNoExtension}": file_stem,
            "${fileExtname}": file_ext,
            "${fileDirname}": file_dir,
            "${fileDirnameBasename}": os.path.basename(file_dir),
            "${relativeFile}": os.path.relpath(file_path, workspace_fs),
            "${relativeFileDirname}": os.path.relpath(file_dir, workspace_fs),
            "${fileWorkspaceFolder}": workspace_fs,
        }

        for token, value in substitutions.items():
            cwd = cwd.replace(token, value)
    elif "${file" in cwd or "${relativeFile" in cwd:
        cwd = workspace_fs

    return cwd


def _get_global_defaults() -> dict[str, Any]:
    return {
        "path": GLOBAL_SETTINGS.get("path", []),
        "interpreter": GLOBAL_SETTINGS.get("interpreter", [sys.executable]),
        "args": GLOBAL_SETTINGS.get("args", []),
        "importStrategy": GLOBAL_SETTINGS.get("importStrategy", "useBundled"),
        "showNotifications": GLOBAL_SETTINGS.get("showNotifications", "off"),
    }


def _update_workspace_settings(settings: list[dict[str, Any]]) -> None:
    if not settings:
        key = os.getcwd()
        WORKSPACE_SETTINGS[key] = {
            "cwd": key,
            "workspaceFS": key,
            "workspace": uris.from_fs_path(key),
            **_get_global_defaults(),
        }
        return

    for setting in settings:
        key = uris.to_fs_path(setting["workspace"])
        WORKSPACE_SETTINGS[key] = {
            "cwd": key,
            **setting,
            "workspaceFS": key,
        }


def log_to_output(
    message: str, msg_type: lsp.MessageType = lsp.MessageType.Log
) -> None:
    """Send backend logs to the VS Code output channel."""
    LSP_SERVER.window_log_message(lsp.LogMessageParams(type=msg_type, message=message))


def log_error(message: str) -> None:
    LSP_SERVER.window_log_message(
        lsp.LogMessageParams(type=lsp.MessageType.Error, message=message)
    )
    if os.getenv("LS_SHOW_NOTIFICATION", "off") in ["onError", "onWarning", "always"]:
        LSP_SERVER.window_show_message(
            lsp.ShowMessageParams(type=lsp.MessageType.Error, message=message)
        )


if __name__ == "__main__":
    LSP_SERVER.start_io()
