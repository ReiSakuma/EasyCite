# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Tests for EasyCite v2 citation detection and local recommendations."""
import pathlib
import sys
import types


def _setup_mocks():
    class _MockLS:
        def __init__(self, **kwargs):
            pass

        def feature(self, *args, **kwargs):
            return lambda f: f

        def window_log_message(self, *args, **kwargs):
            pass

        def window_show_message(self, *args, **kwargs):
            pass

    mock_lsp_server = types.ModuleType("pygls.lsp.server")
    mock_lsp_server.LanguageServer = _MockLS

    mock_lsp_mod = types.ModuleType("pygls.lsp")
    mock_lsp_mod.server = mock_lsp_server

    mock_workspace = types.ModuleType("pygls.workspace")
    mock_workspace.TextDocument = type("TextDocument", (), {"path": None})

    mock_uris = types.ModuleType("pygls.uris")
    mock_uris.from_fs_path = lambda p: "file://" + p
    mock_uris.to_fs_path = lambda u: u.replace("file://", "")

    mock_lsp = types.ModuleType("lsprotocol.types")
    for _name in ["INITIALIZE", "EXIT", "SHUTDOWN"]:
        setattr(mock_lsp, _name, _name)
    for _name in [
        "InitializeParams",
        "LogMessageParams",
        "ShowMessageParams",
    ]:
        setattr(mock_lsp, _name, type(_name, (), {"__init__": lambda self, *args, **kwargs: None}))
    mock_lsp.MessageType = type(
        "MessageType",
        (),
        {"Log": 4, "Error": 1, "Warning": 2, "Info": 3},
    )

    for _mod_name, _mod in [
        ("pygls", types.ModuleType("pygls")),
        ("pygls.lsp", mock_lsp_mod),
        ("pygls.lsp.server", mock_lsp_server),
        ("pygls.workspace", mock_workspace),
        ("pygls.uris", mock_uris),
        ("lsprotocol", types.ModuleType("lsprotocol")),
        ("lsprotocol.types", mock_lsp),
        ("lsp_jsonrpc", types.ModuleType("lsp_jsonrpc")),
    ]:
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = _mod

    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_mocks()

import lsp_server  # noqa: E402


def test_audit_claim_detects_related_work_summary():
    result = lsp_server.audit_claim(
        {
            "claim": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            "existing_citations": [],
        }
    )

    assert result["citation_needed"] is True
    assert result["claim_type"] == "related_work_summary"
    assert result["source"] == "heuristic"
    assert result["suggestions"] == []


def test_audit_claim_does_not_suggest_when_citation_already_present():
    result = lsp_server.audit_claim(
        {
            "claim": "Existing methods use optimization \\cite{smith2024slam}.",
            "existing_citations": ["smith2024slam"],
        }
    )

    assert result["citation_needed"] is False
    assert result["suggestions"] == []
    assert result["source"] == "heuristic"


def test_audit_claim_short_writing_text_does_not_need_citation():
    result = lsp_server.audit_claim(
        {
            "claim": "We discuss limitations.",
            "existing_citations": [],
        }
    )

    assert result["citation_needed"] is False
    assert result["claim_type"] == "writing_only"


def test_audit_claim_rejects_empty_claim():
    try:
        lsp_server.audit_claim({"claim": "   "})
    except ValueError as exc:
        assert "non-empty claim" in str(exc)
    else:
        raise AssertionError("Expected empty claim to raise ValueError.")


def test_audit_claim_lsp_handler_accepts_language_server_argument():
    result = lsp_server.audit_claim_request(
        None,
        {
            "claim": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            "existing_citations": [],
        },
    )

    assert result["citation_needed"] is True
    assert result["claim_type"] == "related_work_summary"


def test_audit_claim_accepts_positional_jsonrpc_params():
    result = lsp_server.audit_claim(
        [
            "",
            "/tmp/easycite-test/cite.tex",
            0,
            0,
            "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            [],
        ]
    )

    assert result["citation_needed"] is True
    assert result["claim_type"] == "related_work_summary"


def test_translate_evidence_lsp_handler_accepts_positional_params():
    result = lsp_server.translate_evidence_request(
        None,
        [
            {
                "texts": [
                    {
                        "id": "claim",
                        "text": "Existing multi-agent Gaussian SLAM methods rely on optimization.",
                    }
                ]
            }
        ],
    )

    assert result["available"] is False
    assert "未配置模型 API" in result["reason"]


def test_translate_evidence_lsp_handler_accepts_raw_text_array():
    result = lsp_server.translate_evidence_request(
        None,
        [
            {
                "id": "claim",
                "text": "Existing multi-agent Gaussian SLAM methods rely on optimization.",
            },
            {
                "id": "reason",
                "text": "The evidence partially supports the claim.",
            },
        ],
    )

    assert result["available"] is False
    assert "未配置模型 API" in result["reason"]


def test_resolve_bibtex_params_accepts_positional_wrapper():
    normalized = lsp_server.normalize_resolve_bibtex_params(
        [
            {
                "citation_keys": ["smith2026localGaussianSlam"],
                "suggestions": [
                    {
                        "citation_key": "smith2026localGaussianSlam",
                        "source": "zotero",
                    }
                ],
            }
        ]
    )

    assert normalized["citation_keys"] == ["smith2026localGaussianSlam"]
    assert normalized["suggestions"][0]["source"] == "zotero"


def test_audit_claim_returns_current_bib_candidates(tmp_path):
    tex_path = tmp_path / "paper.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(
        "\\documentclass{article}\n"
        "\\addbibresource{references.bib}\n"
        "\\begin{document}\n"
        "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    bib_path.write_text(
        "@inproceedings{smith2024slam,\n"
        "  title={Multi-Agent Gaussian SLAM with Offline Global Optimization},\n"
        "  author={Smith, Ada and Zhang, Bo},\n"
        "  booktitle={International Conference on Robotics and Automation},\n"
        "  year={2024},\n"
        "  abstract={This paper studies multi-agent Gaussian SLAM and offline global optimization.}\n"
        "}\n",
        encoding="utf-8",
    )

    result = lsp_server.audit_claim(
        {
            "workspace_path": str(tmp_path),
            "file_path": str(tex_path),
            "claim": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            "existing_citations": [],
        }
    )

    assert result["citation_needed"] is True
    assert result["suggestions"]
    suggestion = result["suggestions"][0]
    assert suggestion["source"] == "current_bib"
    assert suggestion["citation_key"] == "smith2024slam"
    assert suggestion["verdict"] == "direct_support"
    assert "Gaussian SLAM" in suggestion["title"]
