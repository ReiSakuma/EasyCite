from __future__ import annotations

import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL_DIR = os.fsdecode(_PROJECT_ROOT / "bundled" / "tool")
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from citation_agent.tool_health import run_tool_health_check  # noqa: E402


def test_tool_health_returns_core_items(monkeypatch):
    monkeypatch.setenv("CITATION_AGENT_USE_LLM", "false")
    monkeypatch.setenv("CITATION_AGENT_ZOTERO_ENABLED", "false")
    monkeypatch.setenv("CITATION_AGENT_EMBEDDING_ENABLED", "false")
    monkeypatch.setenv("CITATION_AGENT_MEMORY_ENABLED", "true")

    result = run_tool_health_check({
        "workspace_path": os.getcwd(),
        "file_path": "",
    })

    assert result["schema_version"] == "v17.tool_health.1"
    assert any(tool["name"] == "zotero.search" for tool in result["registry"])
    assert result["summary"]["overall_status"] in {"ok", "warning", "error"}
    names = {item["name"] for item in result["items"]}
    assert {
        "python",
        "python_dependencies",
        "llm",
        "zotero",
        "better_bibtex",
        "embedding",
        "pdf_reader",
        "bibtex_workspace",
        "memory",
    }.issubset(names)


def test_tool_health_marks_disabled_optional_tools(monkeypatch):
    monkeypatch.setenv("CITATION_AGENT_USE_LLM", "false")
    monkeypatch.setenv("CITATION_AGENT_ZOTERO_ENABLED", "false")
    monkeypatch.setenv("CITATION_AGENT_EMBEDDING_ENABLED", "false")
    monkeypatch.setenv("CITATION_AGENT_MEMORY_ENABLED", "false")

    result = run_tool_health_check({
        "workspace_path": os.getcwd(),
        "file_path": "",
    })
    by_name = {item["name"]: item for item in result["items"]}

    assert by_name["llm"]["status"] == "disabled"
    assert by_name["zotero"]["status"] == "disabled"
    assert by_name["better_bibtex"]["status"] == "disabled"
    assert by_name["embedding"]["status"] == "disabled"
    assert by_name["memory"]["status"] == "disabled"
