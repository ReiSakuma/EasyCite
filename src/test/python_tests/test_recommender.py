"""Tests for v2 citation recommendation orchestration and Zotero adapter."""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
from unittest.mock import patch


def _setup_path():
    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_path()

from citation_agent.config import ModelConfig, SourceConfig  # noqa: E402
from citation_agent.claim_analysis import build_claim_profile  # noqa: E402
from citation_agent.bibtex_resolver import resolve_bibtex_entries  # noqa: E402
from citation_agent.evidence import (  # noqa: E402
    EvidenceJudge,
    EvidenceRetriever,
    apply_claim_scope_guard,
    cap_confidence_for_verdict,
    is_external_reference_discussion,
    is_low_quality_evidence_passage,
)
from citation_agent.recommender import (  # noqa: E402
    CitationRecommender,
    build_evidence_shortlist,
    dedupe_suggestions,
    filter_excluded_scope_candidates,
    sort_verified_suggestions,
    violates_exclusion_scope,
    verify_evidence,
)
from citation_agent.query_planner import RetrievalQueryPlan, plan_retrieval_queries  # noqa: E402
from citation_agent.schemas import CitationSuggestion, EvidenceSnippet  # noqa: E402
from citation_agent.zotero import ZoteroClient  # noqa: E402


ZOTERO_PAYLOAD = {
    "items": [
        {
            "key": "Z1",
            "title": "Distributed Visual SLAM for Multi-Agent Robots",
            "citationKey": "zotero2025distributed",
            "creators": [
                {"firstName": "Mira", "lastName": "Chen"},
                {"name": "Noah Patel"},
            ],
            "date": "2025-03-04",
            "publicationTitle": "Robotics Letters",
            "abstractNote": "A Zotero candidate about distributed multi-agent visual SLAM.",
            "pdfPath": "/tmp/distributed-visual-slam.pdf",
        }
    ]
}


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _make_urlopen(payload=ZOTERO_PAYLOAD, fail_search=False, bbt_result=None):
    calls: list[str] = []
    rpc_payloads: list[dict] = []

    def _urlopen(request, timeout=0):
        calls.append(request.full_url)
        if "/better-bibtex/json-rpc" in request.full_url:
            rpc_payload = json.loads(request.data.decode("utf-8"))
            rpc_payloads.append(rpc_payload)
            return _FakeResponse(
                {
                    "jsonrpc": "2.0",
                    "result": bbt_result or {},
                    "id": rpc_payload.get("id"),
                }
            )
        if fail_search and "/search?" in request.full_url:
            raise urllib.error.HTTPError(
                request.full_url,
                500,
                "mock search failure",
                hdrs=None,
                fp=None,
            )
        return _FakeResponse(payload)

    return _urlopen, calls, rpc_payloads


def _write_tex_project(tmp_path):
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
    return tex_path


def test_zotero_client_reads_search_endpoint():
    urlopen, calls, _rpc_payloads = _make_urlopen()

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        suggestions = ZoteroClient("http://zotero.local").search(
            "multi-agent visual SLAM", limit=2
        )

    assert len(suggestions) == 1
    assert "/search?" in calls[0]
    assert "q=multi-agent+visual+SLAM" in calls[0]
    assert suggestions[0].source == "zotero"
    assert suggestions[0].citation_key == "zotero2025distributed"
    assert suggestions[0].authors == ["Mira Chen", "Noah Patel"]
    assert suggestions[0].year == "2025"
    assert suggestions[0].pdf_path == "/tmp/distributed-visual-slam.pdf"


def test_zotero_client_falls_back_to_items_endpoint():
    urlopen, calls, _rpc_payloads = _make_urlopen(fail_search=True)

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        suggestions = ZoteroClient("http://zotero.local").search(
            "multi-agent visual SLAM"
        )

    assert len(suggestions) == 1
    assert "/search?" in calls[0]
    assert "/items?" in calls[1]
    assert suggestions[0].source == "zotero"
    assert suggestions[0].citation_key == "zotero2025distributed"


def test_zotero_client_reads_official_local_api_shape():
    local_api_payload = [
        {
            "key": "ABCD1234",
            "data": {
                "key": "ABCD1234",
                "title": "Local Zotero Gaussian SLAM Paper",
                "creators": [{"firstName": "Ada", "lastName": "Smith"}],
                "date": "2026",
                "publicationTitle": "Local Zotero Journal",
                "abstractNote": "A local Zotero API item about Gaussian SLAM.",
            },
        }
    ]
    urlopen, calls, rpc_payloads = _make_urlopen(
        payload=local_api_payload,
        bbt_result={"ABCD1234": "smith2026localGaussianSlam"},
    )

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        suggestions = ZoteroClient("http://127.0.0.1:23119/api").search(
            "Gaussian SLAM", limit=3
        )

    assert "/api/users/0/items/top?" in calls[0]
    assert "/better-bibtex/json-rpc" in calls[1]
    assert "q=Gaussian+SLAM" in calls[0]
    assert rpc_payloads[0]["method"] == "item.citationkey"
    assert rpc_payloads[0]["params"] == [["ABCD1234"]]
    assert suggestions[0].source == "zotero"
    assert suggestions[0].title == "Local Zotero Gaussian SLAM Paper"
    assert suggestions[0].citation_key == "smith2026localGaussianSlam"


def test_zotero_client_falls_back_when_better_bibtex_has_no_key():
    local_api_payload = [
        {
            "key": "NO_KEY",
            "data": {
                "key": "NO_KEY",
                "title": "Local Zotero Paper Without BBT Key",
                "creators": [{"firstName": "Ada", "lastName": "Smith"}],
                "date": "2026",
                "publicationTitle": "Local Zotero Journal",
                "abstractNote": "A local Zotero API item without a BBT citation key.",
            },
        }
    ]
    urlopen, _calls, rpc_payloads = _make_urlopen(
        payload=local_api_payload,
        bbt_result={"NO_KEY": None},
    )

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        suggestions = ZoteroClient("http://127.0.0.1:23119/api").search(
            "Gaussian SLAM", limit=3
        )

    assert rpc_payloads[0]["method"] == "item.citationkey"
    assert suggestions[0].source == "zotero"
    assert suggestions[0].citation_key is None


def test_zotero_client_exports_bibtex_with_better_bibtex():
    exported = "@article{smith2026localGaussianSlam,\n  title = {Local Zotero Gaussian SLAM Paper}\n}"
    urlopen, calls, rpc_payloads = _make_urlopen(bbt_result=exported)

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        bibtex = ZoteroClient("http://127.0.0.1:23119/api").export_bibtex(
            ["smith2026localGaussianSlam"]
        )

    assert "/better-bibtex/json-rpc" in calls[0]
    assert rpc_payloads[0]["method"] == "item.export"
    assert rpc_payloads[0]["params"] == [["smith2026localGaussianSlam"], "Better BibTeX"]
    assert bibtex == exported


def test_resolve_bibtex_entries_exports_zotero_and_fallback_metadata():
    exported = "@article{smith2026localGaussianSlam,\n  title = {Local Zotero Gaussian SLAM Paper}\n}"
    urlopen, _calls, rpc_payloads = _make_urlopen(bbt_result=exported)

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        result = resolve_bibtex_entries(
            {
                "citation_keys": ["smith2026localGaussianSlam", "alreadyInBib"],
                "suggestions": [
                    {
                        "citation_key": "smith2026localGaussianSlam",
                        "source": "zotero",
                    },
                    {
                        "citation_key": "alreadyInBib",
                        "source": "current_bib",
                        "title": "Existing Bib Candidate",
                        "authors": ["Ada Smith", "Bo Zhang"],
                        "year": "2024",
                        "venue": "Journal of Local Robotics",
                        "doi": "10.1000/local.2024",
                    },
                ],
            },
            SourceConfig(
                max_suggestions=3,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
            ),
        )

    assert rpc_payloads[0]["method"] == "item.export"
    assert len(result["entries"]) == 2
    assert result["entries"][0]["citation_key"] == "smith2026localGaussianSlam"
    assert result["entries"][0]["bibtex"] == exported
    assert result["entries"][0]["source"] == "better-bibtex:zotero"
    assert result["entries"][0]["source_trust"] == "high"
    assert result["entries"][0]["source_rank"] == 1
    assert result["entries"][0]["requires_user_verification"] is False
    assert result["entries"][1]["citation_key"] == "alreadyInBib"
    assert result["entries"][1]["bibtex"] == (
        "@article{alreadyInBib,\n"
        "  title = {Existing Bib Candidate},\n"
        "  author = {Ada Smith and Bo Zhang},\n"
        "  year = {2024},\n"
        "  journal = {Journal of Local Robotics},\n"
        "  doi = {10.1000/local.2024},\n"
        "  note = {Candidate-metadata fallback generated by EasyCite; verify before submission.}\n"
        "}"
    )
    assert result["entries"][1]["source"] == "candidate-metadata-fallback:current_bib"
    assert result["entries"][1]["source_trust"] == "low"
    assert result["entries"][1]["source_rank"] == 4
    assert result["entries"][1]["requires_user_verification"] is True
    assert result["entries"][1]["quality_warnings"]
    assert result["missing"] == []


def test_resolve_bibtex_entries_exports_bbt_key_without_suggestion_metadata():
    exported = "@article{zoteroKey,\n  title = {Zotero Paper}\n}"
    urlopen, _calls, rpc_payloads = _make_urlopen(bbt_result=exported)

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        result = resolve_bibtex_entries(
            {
                "citation_keys": ["zoteroKey"],
                "suggestions": [],
            },
            SourceConfig(
                max_suggestions=3,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
            ),
        )

    assert rpc_payloads[0]["method"] == "item.export"
    assert rpc_payloads[0]["params"] == [["zoteroKey"], "Better BibTeX"]
    assert len(result["entries"]) == 1
    assert result["entries"][0]["citation_key"] == "zoteroKey"
    assert result["entries"][0]["bibtex"] == exported
    assert result["entries"][0]["source"] == "better-bibtex:zotero"
    assert result["entries"][0]["source_trust"] == "high"
    assert result["entries"][0]["requires_user_verification"] is False
    assert result["missing"] == []


def test_resolve_bibtex_entries_rejects_mismatched_bbt_export_key():
    exported = "@article{differentKey,\n  title = {Wrong Paper}\n}"
    urlopen, _calls, rpc_payloads = _make_urlopen(bbt_result=exported)

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        result = resolve_bibtex_entries(
            {
                "citation_keys": ["zoteroKey"],
                "suggestions": [
                    {
                        "citation_key": "zoteroKey",
                        "source": "zotero",
                    }
                ],
            },
            SourceConfig(
                max_suggestions=3,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
            ),
        )

    assert rpc_payloads[0]["method"] == "item.export"
    assert result["entries"] == []
    assert result["missing"][0]["citation_key"] == "zoteroKey"


def test_resolve_bibtex_entries_reports_missing_when_metadata_is_insufficient():
    result = resolve_bibtex_entries(
        {
            "citation_keys": ["unknownKey"],
            "suggestions": [
                {
                    "citation_key": "unknownKey",
                    "source": "online",
                }
            ],
        },
        SourceConfig(max_suggestions=3, zotero_enabled=False, zotero_endpoint=None),
    )

    assert result["entries"] == []
    assert result["missing"] == [
        {
            "citation_key": "unknownKey",
            "reason": "Zotero 未启用或 endpoint 为空，且候选元数据不足，无法生成兜底 BibTeX。",
        }
    ]


def test_resolve_bibtex_entries_treats_zotero_paper_id_as_zotero_source():
    exported = "@article{zoteroKey,\n  title = {Zotero Paper}\n}"
    urlopen, _calls, rpc_payloads = _make_urlopen(bbt_result=exported)

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        result = resolve_bibtex_entries(
            {
                "citation_keys": ["zoteroKey"],
                "suggestions": [
                    {
                        "citation_key": "zoteroKey",
                        "source": "unknown",
                        "paper_id": "zotero:ABCD1234",
                        "title": "Zotero Paper",
                    }
                ],
            },
            SourceConfig(
                max_suggestions=3,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
            ),
        )

    assert rpc_payloads[0]["method"] == "item.export"
    assert len(result["entries"]) == 1
    assert result["entries"][0]["citation_key"] == "zoteroKey"
    assert result["entries"][0]["bibtex"] == exported
    assert result["entries"][0]["source"] == "better-bibtex:zotero"
    assert result["entries"][0]["source_trust"] == "high"
    assert result["missing"] == []


def test_recommender_orchestrates_bib_then_zotero(tmp_path):
    tex_path = _write_tex_project(tmp_path)
    urlopen, _calls, _rpc_payloads = _make_urlopen()

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        recommender = CitationRecommender(
            SourceConfig(
                max_suggestions=3,
                zotero_enabled=True,
                zotero_endpoint="http://zotero.local",
            )
        )
        suggestions = recommender.recommend(
            {
                "workspace_path": str(tmp_path),
                "file_path": str(tex_path),
                "claim": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            }
        )

    assert [suggestion.source for suggestion in suggestions] == ["current_bib", "zotero"]
    assert suggestions[0].citation_key == "smith2024slam"
    assert suggestions[1].citation_key == "zotero2025distributed"


def test_recommender_deduplicates_zotero_candidate_already_in_bib(tmp_path):
    tex_path = _write_tex_project(tmp_path)
    duplicate_payload = {
        "items": [
            {
                "key": "Z2",
                "title": "Duplicate Candidate",
                "citationKey": "smith2024slam",
            }
        ]
    }
    urlopen, _calls, _rpc_payloads = _make_urlopen(payload=duplicate_payload)

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        recommender = CitationRecommender(
            SourceConfig(
                max_suggestions=3,
                zotero_enabled=True,
                zotero_endpoint="http://zotero.local",
            )
        )
        suggestions = recommender.recommend(
            {
                "workspace_path": str(tmp_path),
                "file_path": str(tex_path),
                "claim": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            }
        )

    assert len(suggestions) == 1
    assert suggestions[0].source == "current_bib"
    assert suggestions[0].citation_key == "smith2024slam"


def test_dedupe_suggestions_merges_duplicate_titles_with_different_citekeys():
    first = CitationSuggestion(
        paper_id="zotero:A",
        title="GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM",
        citation_key="thomasGRANDSLAMLocalOptimization2025",
        confidence=0.68,
        source="zotero",
        reason="first",
        evidence=EvidenceSnippet(text="abstract", source="abstract"),
    )
    second = first.model_copy(
        update={
            "paper_id": "zotero:B",
            "citation_key": "thomasGRANDSLAMLocalOptimization2025a",
            "confidence": 0.7,
        }
    )

    deduped = dedupe_suggestions([first, second])

    assert len(deduped) == 1
    assert deduped[0].title == first.title


def test_dedupe_suggestions_merges_near_duplicate_titles_with_generic_suffix():
    first = CitationSuggestion(
        paper_id="zotero:A",
        title="Collaborative Neural Point-Based Mapping",
        citation_key="smithCollaborativeNeuralMapping2024",
        confidence=0.68,
        source="zotero",
        reason="first",
        evidence=EvidenceSnippet(text="abstract", source="abstract"),
    )
    second = first.model_copy(
        update={
            "paper_id": "zotero:B",
            "title": "Collaborative Neural Point-Based Mapping System",
            "citation_key": "smithCollaborativeNeuralMappingSystem2024",
            "confidence": 0.7,
        }
    )

    deduped = dedupe_suggestions([first, second])

    assert len(deduped) == 1
    assert deduped[0].citation_key == "smithCollaborativeNeuralMappingSystem2024"


def test_verified_suggestion_sorting_is_stable_for_ties():
    alpha = CitationSuggestion(
        paper_id="zotero:B",
        title="Beta Gaussian SLAM",
        citation_key="beta2025",
        confidence=0.68,
        source="zotero",
        verdict="partial_support",
        reason="tie",
        evidence=EvidenceSnippet(text="tie", source="abstract"),
    )
    beta = alpha.model_copy(
        update={
            "paper_id": "zotero:A",
            "title": "Alpha Gaussian SLAM",
            "citation_key": "alpha2025",
        }
    )

    assert sort_verified_suggestions([alpha, beta]) == sort_verified_suggestions([beta, alpha])
    assert sort_verified_suggestions([alpha, beta])[0].title == "Alpha Gaussian SLAM"


def test_external_reference_discussion_is_not_own_paper_evidence():
    grand_slam = CitationSuggestion(
        paper_id="zotero:GRAND",
        title="GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM",
        citation_key="thomasGRANDSLAMLocalOptimization2025",
        confidence=0.68,
        source="zotero",
        reason="candidate",
        evidence=EvidenceSnippet(text="abstract", source="abstract"),
    )

    assert is_external_reference_discussion(
        "MAGIC-SLAM [38] extends 3DGS SLAM to support loop closure by leveraging a pretrained vision-language model.",
        grand_slam,
    )
    assert not is_external_reference_discussion(
        "This paper presents GRAND-SLAM, a local optimization approach for large-scale multi-agent Gaussian SLAM.",
        grand_slam,
    )


def test_citation_heavy_related_work_passage_is_not_candidate_own_evidence():
    gaussian_slam = CitationSuggestion(
        paper_id="zotero:GAUSSIAN-SLAM",
        title="Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting",
        citation_key="yugayGaussianSLAMPhotorealisticDense",
        confidence=0.9,
        source="zotero",
        reason="candidate",
        evidence=EvidenceSnippet(text="abstract", source="abstract"),
    )

    passage = (
        "Inspired by Neural Radiance Fields [40], there has been immense progress "
        "in dense surface reconstruction [47,67] and accurate pose estimation "
        "[2, 29, 52, 69]. These efforts have led to the development of comprehensive "
        "dense SLAM systems [34, 54, 63, 78, 85, 88, 89], showing a trend in the "
        "pursuit of precise and reliable visual SLAM. A comprehensive survey on "
        "online RGBD reconstruction can be found in [90]. While the latest neural "
        "methods show impressive rendering capabilities on synthetic data, they "
        "struggle when applied to real-world data."
    )

    assert is_external_reference_discussion(passage, gaussian_slam)


def test_low_quality_evidence_filter_rejects_reference_entries_and_diagram_text():
    diagram_text = (
        "AGENT SIDE Agent i j k Tracking & Local Mapping Loop Closure Detection "
        "3D Gaussian Map Representation Pose Optimization & Map Refinement Data Stream "
        "SERVER SIDE Inter-Agent Loop Closure Semantic Extraction Sub-map Merging & "
        "Global Map Refinement Pose Graph Optimization MULTI-AGENT SYSTEM Agent i j k "
        "Data Stream Tracking & Local Mapping Loop Closure & Detection Fig. 3. "
        "Core Components of CoGS-SLAM: Centralized (left), Distributed (right)."
    )
    reference_text = (
        "Grand-slam: Local optimization for globally consistent large-scale multi-agent "
        "gaussian slam. arXiv preprint arXiv:2506.18885. Tian, Y., Chang, Y., Arias, F.H., "
        "Nieto-Granda, C., How, J.P., and Carlone, L. (2022). Kimera-multi: Robust, "
        "distributed, dense metric-semantic slam for multi-robot systems. IEEE Transactions "
        "on Robotics, 38(4)."
    )
    normal_text = (
        "This work studies multi-agent Gaussian SLAM and uses offline global optimization "
        "to maintain globally consistent maps across multiple collaborating agents."
    )

    assert is_low_quality_evidence_passage(diagram_text)
    assert is_low_quality_evidence_passage(reference_text)
    assert not is_low_quality_evidence_passage(normal_text)


def test_evidence_retrieval_uses_stable_passage_order_for_ties():
    profile = build_claim_profile(
        "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization."
    )
    suggestion = CitationSuggestion(
        paper_id="zotero:TIE",
        title="Tie-Break Gaussian SLAM",
        citation_key="tie2025",
        confidence=0.5,
        source="zotero",
        reason="tie",
        evidence=EvidenceSnippet(
            text=(
                "Gaussian SLAM optimization appears in the first passage. "
                "Gaussian SLAM optimization appears in the second passage."
            ),
            source="abstract",
            section="Zotero abstract",
        ),
    )

    passages = EvidenceRetriever(max_passages=2).retrieve(suggestion, profile)

    assert [passage.source for passage in passages] == ["abstract", "metadata"]


def test_insufficient_evidence_confidence_is_capped_low():
    assert cap_confidence_for_verdict(1.0, "insufficient_evidence") == 0.35
    assert cap_confidence_for_verdict(0.95, "contradiction") == 0.25
    assert cap_confidence_for_verdict(0.9, "direct_support") == 0.9


def test_claim_profile_generates_short_zotero_queries():
    profile = build_claim_profile(
        "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization."
    )

    joined_queries = " ".join(profile.query_phrases).lower()
    assert "multi-agent gaussian slam" in joined_queries
    assert "offline global optimization" in joined_queries
    assert "offline" in profile.support_terms
    assert "optimization" in profile.support_terms


def test_neural_representation_claim_does_not_query_gaussian_from_context():
    profile = build_claim_profile(
        "Recent neural and differentiable representations have changed this design space "
        "by treating mapping as a joint problem of geometry estimation, appearance modeling, "
        "and camera pose refinement.",
        context=(
            "3D Gaussian Splatting is especially appealing for robotics because it represents "
            "a scene with explicit Gaussian primitives."
        ),
    )

    joined_queries = " ".join(profile.query_phrases).lower()
    assert profile.claim_kind == "field_level_background"
    assert profile.retrieval_intent == "prefer_survey_or_foundational_background"
    assert "gaussian" in profile.exclude_terms
    assert "neural and differentiable representations" in joined_queries
    assert "geometry" in joined_queries
    assert "gaussian slam" not in joined_queries
    assert "gaussian splatting" not in profile.phrases


def test_scope_guard_downgrades_gaussian_system_for_broad_neural_claim():
    suggestion = CitationSuggestion(
        paper_id="zotero:GS",
        title="GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting",
        citation_key="yanGSSLAMDenseVisual2024",
        confidence=0.95,
        source="zotero",
        reason="candidate",
        evidence=EvidenceSnippet(text="abstract", source="abstract"),
    )
    passages = [
        EvidenceSnippet(
            source="abstract",
            section="Abstract",
            text=(
                "In this paper, we introduce GS-SLAM that first utilizes 3D Gaussian "
                "representation in the Simultaneous Localization and Mapping system."
            ),
            score=0.9,
        )
    ]

    verdict, confidence, reason = apply_claim_scope_guard(
        claim=(
            "Recent neural and differentiable representations have changed this design space "
            "by treating mapping as a joint problem of geometry estimation, appearance modeling, "
            "and camera pose refinement."
        ),
        suggestion=suggestion,
        passages=passages,
        verdict="direct_support",
        confidence=0.95,
        reason="LLM judged direct support.",
    )

    assert verdict == "insufficient_evidence"
    assert confidence <= 0.35
    assert "作用域校验" in reason


def test_scope_guard_downgrades_non_foundational_system_verdict_for_field_background():
    claim = (
        "Recent neural and differentiable representations have changed this design space "
        "by treating mapping as a joint problem of geometry estimation, appearance modeling, "
        "and camera pose refinement."
    )
    profile = build_claim_profile(claim)
    profile = profile.__class__(
        **{
            **profile.__dict__,
            "required_concept_terms": {"neural", "differentiable", "representation"},
            "foundational_concept_terms": {"implicit", "rendering"},
        }
    )
    suggestion = CitationSuggestion(
        paper_id="zotero:APP",
        title="A Neural Differentiable Mapping System for Mobile Robots",
        citation_key="smithNeuralDifferentiableMappingSystem2024",
        confidence=0.9,
        source="zotero",
        reason="candidate",
        evidence=EvidenceSnippet(text="abstract", source="abstract"),
    )
    passages = [
        EvidenceSnippet(
            source="abstract",
            section="Abstract",
            text=(
                "This application system uses neural differentiable mapping and implicit rendering "
                "inside a robot pipeline."
            ),
            score=0.9,
        )
    ]

    verdict, confidence, reason = apply_claim_scope_guard(
        claim=claim,
        suggestion=suggestion,
        passages=passages,
        verdict="foundational_support",
        confidence=0.86,
        reason="LLM judged foundational support.",
        profile=profile,
    )

    assert verdict == "partial_support"
    assert confidence <= 0.62
    assert "奠基性校验" in reason


def test_scope_filter_removes_gaussian_candidates_for_neural_representation_claim():
    profile = build_claim_profile(
        (
            "Recent neural and differentiable representations have changed this design space "
            "by treating mapping as a joint problem of geometry estimation, appearance modeling, "
            "and camera pose refinement."
        ),
        context=(
            "3D Gaussian Splatting is especially appealing for robotics because it represents "
            "a scene with explicit Gaussian primitives."
        ),
    )
    gaussian = CitationSuggestion(
        paper_id="zotero:GAUSSIAN",
        title="Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting",
        citation_key="yugayGaussianSLAMPhotorealisticDense",
        confidence=0.2,
        source="zotero",
        reason="topic related",
        evidence=EvidenceSnippet(text="neural representations rely on pose updates", source="pdf"),
    )
    neural = CitationSuggestion(
        paper_id="zotero:NEURAL",
        title="Neural Implicit Mapping for Dense Visual SLAM",
        citation_key="smithNeuralImplicitMapping2024",
        confidence=0.7,
        source="zotero",
        reason="candidate",
        evidence=EvidenceSnippet(text="neural implicit maps jointly optimize geometry, appearance, and pose", source="abstract"),
    )

    assert violates_exclusion_scope(gaussian, profile)
    assert not violates_exclusion_scope(neural, profile)
    assert filter_excluded_scope_candidates([gaussian, neural], profile) == [neural]


def test_scope_filter_uses_context_only_terms_without_domain_hardcoding():
    profile = build_claim_profile(
        "Protein language models encode residue-level representations for fold recognition.",
        context=(
            "Graph neural networks are also widely used for molecular property prediction "
            "in computational biology."
        ),
    )
    graph_candidate = CitationSuggestion(
        paper_id="zotero:GRAPH",
        title="Graph Neural Networks for Molecular Property Prediction",
        citation_key="leeGraphNeuralNetworks2024",
        confidence=0.4,
        source="zotero",
        reason="topic related",
        evidence=EvidenceSnippet(text="graph models predict molecules", source="abstract"),
    )
    protein_candidate = CitationSuggestion(
        paper_id="zotero:PROTEIN",
        title="Protein Language Models for Structure Prediction",
        citation_key="smithProteinLanguageModels2024",
        confidence=0.7,
        source="zotero",
        reason="candidate",
        evidence=EvidenceSnippet(
            text="protein language models encode residue representations for fold recognition",
            source="abstract",
        ),
    )

    assert {"graph", "molecular"} <= profile.exclude_terms
    assert violates_exclusion_scope(graph_candidate, profile)
    assert not violates_exclusion_scope(protein_candidate, profile)


def test_broad_field_claim_keeps_related_system_paper_as_diagnostic(tmp_path):
    payload = [
        {
            "key": "GS",
            "data": {
                "key": "GS",
                "itemType": "journalArticle",
                "title": "GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting",
                "date": "2024",
                "citationKey": "yanGSSLAMDenseVisual2024",
                "abstractNote": (
                    "The paper presents a method that jointly optimizes geometry, appearance, "
                    "and camera pose with differentiable rendering, but it does not assert that "
                    "recent representations changed the broader design space."
                ),
            },
        },
    ]
    urlopen, _calls, _rpc_payloads = _make_urlopen(
        payload=payload,
        bbt_result={"GS": "yanGSSLAMDenseVisual2024"},
    )

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        bundle = CitationRecommender(
            SourceConfig(
                max_suggestions=5,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
                max_metadata_candidates=10,
                max_abstract_candidates=5,
                max_evidence_passages=3,
            )
        ).recommend_with_diagnostics(
            {
                "workspace_path": str(tmp_path),
                "file_path": str(tmp_path / "paper.tex"),
                "claim": (
                    "Recent neural and differentiable representations have changed this design space "
                    "by treating mapping as a joint problem of geometry estimation, appearance modeling, "
                    "and camera pose refinement."
                ),
            }
        )

    assert bundle.suggestions == []
    assert [suggestion.citation_key for suggestion in bundle.diagnostic_suggestions] == [
        "yanGSSLAMDenseVisual2024"
    ]


def test_recommender_persists_v17_langgraph_agent_run_record(tmp_path):
    tex_path = _write_tex_project(tmp_path)

    bundle = CitationRecommender(
        SourceConfig(
            max_suggestions=3,
            zotero_enabled=False,
            zotero_endpoint=None,
            max_metadata_candidates=10,
            max_abstract_candidates=3,
            max_evidence_passages=2,
            agent_runtime="langgraph",
        )
    ).recommend_with_diagnostics(
        {
            "workspace_path": str(tmp_path),
            "file_path": str(tex_path),
            "line_start": 3,
            "line_end": 3,
            "claim": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            "context": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            "existing_citations": [],
        }
    )

    assert bundle.agent_state["schema_version"] == "v20.agent_state.1"
    assert bundle.agent_state["context_engineering"]["schema_version"] == "v16.context.1"
    assert bundle.agent_state["tool_reliability"]["schema_version"] == "v17.tools.1"
    assert bundle.agent_state["request"]["file_path"] == str(tex_path)
    assert bundle.agent_state["candidate_pool"]["counts"]["current_bib_candidates"] == 1
    assert bundle.agent_state["graph"]["schema_version"] == "v9.graph.1"
    assert bundle.agent_state["graph"]["framework"] == "langgraph"
    assert bundle.agent_state["graph"]["runtime"] == "langgraph-stategraph"
    assert bundle.agent_state["graph"]["compiled"] is True
    assert any(event.stage == "graph_runtime" for event in bundle.trace_events)
    assert any(node["name"] == "analyze_claim" for node in bundle.agent_state["graph"]["nodes"])
    assert bundle.agent_state["graph"]["edges"]
    assert bundle.agent_run_path
    run_path = pathlib.Path(bundle.agent_run_path)
    assert run_path.exists()
    saved = json.loads(run_path.read_text(encoding="utf-8"))
    assert saved["run_id"] == bundle.trace_events[0].run_id
    assert saved["output"]["recommendation_count"] == len(bundle.suggestions)
    assert saved["graph"]["checkpoint_policy"]["storage"] == ".citation-agent/runs/{run_id}.json"
    latest = run_path.parent / "latest.json"
    assert latest.exists()


def test_evidence_shortlist_adds_low_rank_contract_hits_without_hardcoded_targets():
    claim = (
        "Recent neural and differentiable representations have changed this design space "
        "by treating mapping as a joint problem of geometry estimation, appearance modeling, "
        "and camera pose refinement."
    )
    profile = build_claim_profile(claim)
    profile = profile.__class__(
        **{
            **profile.__dict__,
            "required_concept_terms": {"neural", "differentiable", "representation", "rendering"},
            "foundational_concept_terms": {"implicit", "field"},
            "retrieval_terms": {"neural", "differentiable", "rendering", "field"},
        }
    )
    top_ranked = [
        CitationSuggestion(
            paper_id=f"zotero:TOP{index}",
            title=f"Top Ranked Context Paper {index}",
            citation_key=f"top{index}",
            confidence=0.7,
            source="zotero",
            reason="metadata high",
            evidence=EvidenceSnippet(text="robot mapping and pose optimization", source="abstract"),
        )
        for index in range(8)
    ]
    weak_tail = CitationSuggestion(
        paper_id="zotero:WEAK",
        title="A Generic Mapping System",
        citation_key="weakTail",
        confidence=0.6,
        source="zotero",
        reason="tail",
        evidence=EvidenceSnippet(text="mapping pose system", source="abstract"),
    )
    strong_tail = CitationSuggestion(
        paper_id="zotero:STRONG",
        title="Foundational Differentiable Neural Field Representation",
        citation_key="strongTail",
        confidence=0.6,
        source="zotero",
        reason="tail",
        evidence=EvidenceSnippet(
            text=(
                "This work introduces a neural implicit field representation optimized "
                "through differentiable rendering."
            ),
            source="abstract",
        ),
    )

    shortlist = build_evidence_shortlist(
        ranked=[*top_ranked, weak_tail, strong_tail],
        profile=profile,
        abstract_limit=8,
        max_extra=2,
    )

    keys = [suggestion.citation_key for suggestion in shortlist]
    assert "strongTail" in keys
    assert "weakTail" not in keys


def test_query_planner_reports_llm_cache_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    calls: list[str] = []

    def fake_llm_plan(**_kwargs):
        calls.append("remote")
        return RetrievalQueryPlan(
            queries=["neural radiance fields"],
            concepts=["neural radiance fields"],
            required_concepts=["differentiable rendering"],
            foundational_concepts=["NeRF"],
            claim_focus="Neural representation claim",
            reason="remote planner",
            source="llm",
        )

    monkeypatch.setattr(
        "citation_agent.query_planner.call_litellm_query_planner",
        fake_llm_plan,
    )
    config = ModelConfig(
        use_llm=True,
        provider="deepseek",
        model="deepseek/test",
        api_key="test-key",
        api_base=None,
    )

    first = plan_retrieval_queries(
        claim="Neural radiance fields changed differentiable scene representation.",
        context="Neural radiance fields changed differentiable scene representation.",
        base_queries=["neural representation"],
        model_config=config,
    )
    second = plan_retrieval_queries(
        claim="Neural radiance fields changed differentiable scene representation.",
        context="Neural radiance fields changed differentiable scene representation.",
        base_queries=["neural representation"],
        model_config=config,
    )
    third = plan_retrieval_queries(
        claim="Neural radiance fields changed differentiable scene representation.",
        context="Neural radiance fields changed differentiable scene representation.",
        base_queries=["neural representation"],
        model_config=config,
        force_refresh=True,
    )

    assert calls == ["remote", "remote"]
    assert first.source == "llm"
    assert second.source == "llm_cache"
    assert "cache hit" in second.reason
    assert third.source == "llm"


def test_recommender_keeps_llm_planned_foundational_query_results_for_review(tmp_path, monkeypatch):
    claim = (
        "Recent neural and differentiable representations have changed this design space "
        "by treating mapping as a joint problem of geometry estimation, appearance modeling, "
        "and camera pose refinement."
    )
    nerf_payload = [
        {
            "key": "NERF",
            "data": {
                "key": "NERF",
                "itemType": "journalArticle",
                "title": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
                "date": "2020",
                "citationKey": "mildenhallNeRFRepresentingScenes2020",
                "abstractNote": (
                    "We represent a continuous scene as a neural radiance field and optimize "
                    "the representation using differentiable volume rendering from posed images."
                ),
            },
        }
    ]
    empty_payload: list[dict] = []
    calls: list[str] = []

    def _urlopen(request, timeout=0):
        calls.append(request.full_url)
        if "/better-bibtex/json-rpc" in request.full_url:
            rpc_payload = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "jsonrpc": "2.0",
                    "result": {"NERF": "mildenhallNeRFRepresentingScenes2020"},
                    "id": rpc_payload.get("id"),
                }
            )
        if "neural+radiance+fields" in request.full_url:
            return _FakeResponse(nerf_payload)
        return _FakeResponse(empty_payload)

    def _fake_plan(**_kwargs):
        return RetrievalQueryPlan(
            queries=["neural radiance fields", "neural implicit representations"],
            concepts=["neural radiance fields", "differentiable volume rendering"],
            required_concepts=["neural radiance fields", "differentiable rendering"],
            foundational_concepts=["NeRF", "neural implicit representation"],
            excluded_concepts=["SLAM", "mapping"],
            claim_focus=(
                "Neural or differentiable scene representations changed visual mapping by "
                "jointly optimizing geometry, appearance, and pose."
            ),
            reason="Foundational neural representation terms.",
            source="llm",
        )

    monkeypatch.setattr("citation_agent.recommender.plan_retrieval_queries", _fake_plan)
    with patch("citation_agent.zotero.urllib.request.urlopen", _urlopen):
        bundle = CitationRecommender(
            SourceConfig(
                max_suggestions=5,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
                max_metadata_candidates=10,
                max_abstract_candidates=5,
                max_evidence_passages=3,
            )
        ).recommend_with_diagnostics(
            {
                "workspace_path": str(tmp_path),
                "file_path": str(tmp_path / "paper.tex"),
                "claim": claim,
            }
        )

    assert [suggestion.citation_key for suggestion in bundle.suggestions] == [
        "mildenhallNeRFRepresentingScenes2020"
    ]
    assert bundle.suggestions[0].verdict == "foundational_support"
    assert any("q=neural+radiance+fields" in call for call in calls)


def test_debug_target_probe_traces_known_paper_when_normal_retrieval_misses(tmp_path, monkeypatch):
    claim = "A recent differentiable mapping method jointly optimizes geometry and camera pose."
    target_payload = [
        {
            "key": "TARGET",
            "data": {
                "key": "TARGET",
                "itemType": "journalArticle",
                "title": "NeRF-SLAM: Neural Radiance Field Mapping with Camera Pose Optimization",
                "date": "2022",
                "citationKey": "wangNeRFSlamNeural2022",
                "abstractNote": (
                    "The method optimizes a neural radiance field map together with camera poses "
                    "using differentiable rendering."
                ),
            },
        }
    ]
    calls: list[str] = []

    def _urlopen(request, timeout=0):
        calls.append(request.full_url)
        if "/better-bibtex/json-rpc" in request.full_url:
            rpc_payload = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "jsonrpc": "2.0",
                    "result": {"TARGET": "wangNeRFSlamNeural2022"},
                    "id": rpc_payload.get("id"),
                }
            )
        if "NeRF-SLAM" in request.full_url or "wangNeRFSlamNeural2022" in request.full_url:
            return _FakeResponse(target_payload)
        return _FakeResponse([])

    monkeypatch.setattr(
        "citation_agent.recommender.plan_retrieval_queries",
        lambda **_kwargs: RetrievalQueryPlan(
            queries=["differentiable mapping"],
            concepts=["differentiable mapping"],
            required_concepts=["differentiable mapping"],
            reason="local only",
            source="local",
        ),
    )
    with patch("citation_agent.zotero.urllib.request.urlopen", _urlopen):
        bundle = CitationRecommender(
            SourceConfig(
                max_suggestions=5,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
                max_metadata_candidates=10,
                max_abstract_candidates=5,
                max_evidence_passages=3,
                debug_target_papers=["NeRF-SLAM"],
            )
        ).recommend_with_diagnostics(
            {
                "workspace_path": str(tmp_path),
                "file_path": str(tmp_path / "paper.tex"),
                "claim": claim,
            }
        )

    trace_text = "\n".join(bundle.target_trace)
    assert "zotero_retrieval_queries: not found" in trace_text
    assert "zotero_forced_target_probe: FOUND" in trace_text
    assert "NeRF-SLAM" in trace_text
    assert any("q=NeRF-SLAM" in call for call in calls)


def test_debug_target_probe_is_diagnostic_only_and_does_not_enter_evidence(tmp_path, monkeypatch):
    target = CitationSuggestion(
        paper_id="zotero:CP",
        title="CP-SLAM: Collaborative Neural Point-based SLAM",
        citation_key="huCPSLAMCollaborativeNeural2023",
        confidence=0.68,
        source="zotero",
        reason="target",
        evidence=EvidenceSnippet(text="Collaborative neural point-based SLAM.", source="abstract"),
    )
    filler = [
        CitationSuggestion(
            paper_id=f"zotero:FILLER{index}",
            title=f"Filler Neural Mapping Paper {index}",
            citation_key=f"filler{index}",
            confidence=0.7,
            source="zotero",
            reason="filler",
            evidence=EvidenceSnippet(text="neural mapping geometry pose", source="abstract"),
        )
        for index in range(12)
    ]

    monkeypatch.setattr(
        "citation_agent.recommender.plan_retrieval_queries",
        lambda **_kwargs: RetrievalQueryPlan(
            queries=["neural mapping"],
            concepts=["neural mapping"],
            reason="test",
            source="local",
        ),
    )
    monkeypatch.setattr(
        "citation_agent.recommender.BibRepository.search",
        lambda *_args, **_kwargs: [],
    )
    def _fake_zotero_search(_self, claim, limit=10, query_phrases=None, metadata_limit=None):  # noqa: ARG001
        if query_phrases == ["CP-SLAM"]:
            return [target]
        return filler

    monkeypatch.setattr("citation_agent.recommender.ZoteroClient.search", _fake_zotero_search)

    seen_by_evidence: list[str] = []

    def _fake_verify(claim, profile, suggestions, max_passages, logger=None, **_kwargs):
        seen_by_evidence.extend(suggestion.citation_key or "" for suggestion in suggestions)
        return [
            suggestion.model_copy(
                update={
                    "verdict": "insufficient_evidence",
                    "confidence": 0.25,
                    "reason": "test evidence result",
                }
            )
            for suggestion in suggestions
        ]

    monkeypatch.setattr("citation_agent.recommender.verify_evidence", _fake_verify)

    bundle = CitationRecommender(
        SourceConfig(
            max_suggestions=5,
            zotero_enabled=True,
            zotero_endpoint="http://127.0.0.1:23119/api",
            max_metadata_candidates=20,
            max_abstract_candidates=8,
            max_evidence_passages=3,
            debug_target_papers=["CP-SLAM"],
        )
    ).recommend_with_diagnostics(
        {
            "workspace_path": str(tmp_path),
            "file_path": str(tmp_path / "paper.tex"),
            "claim": "Recent neural mapping methods jointly optimize geometry and pose.",
        }
    )

    assert "huCPSLAMCollaborativeNeural2023" not in seen_by_evidence
    assert any(
        suggestion.citation_key == "huCPSLAMCollaborativeNeural2023"
        for suggestion in bundle.diagnostic_suggestions
    )
    trace_text = "\n".join(bundle.target_trace)
    assert "zotero_forced_target_probe: FOUND" in trace_text
    assert "probe_only=true" in trace_text
    assert "evidence_shortlist: not found" in trace_text


def test_zotero_local_api_uses_claim_aware_queries_and_filters_attachments():
    local_api_payload = [
        {
            "key": "PAPER1",
            "data": {
                "key": "PAPER1",
                "itemType": "journalArticle",
                "title": "Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting",
                "date": "2024",
                "citationKey": "yugayGaussianSLAMPhotorealisticDense",
                "abstractNote": "This paper studies dense SLAM with Gaussian splatting.",
            },
            "links": {
                "attachment": {
                    "href": "http://127.0.0.1:23119/api/users/0/items/PAPER1/children"
                }
            },
        },
        {
            "key": "ATTACH1",
            "data": {
                "key": "ATTACH1",
                "itemType": "attachment",
                "title": "PDF",
            },
        },
    ]
    attachment_payload = [
        {
            "key": "PDF1",
            "data": {"key": "PDF1", "itemType": "attachment", "title": "PDF"},
            "links": {
                "enclosure": {
                    "href": "file:///C:/Users/tester/Zotero/storage/PDF1/paper.pdf"
                }
            },
        }
    ]
    calls: list[str] = []
    rpc_payloads: list[dict] = []

    def _urlopen(request, timeout=0):
        calls.append(request.full_url)
        if request.full_url.endswith("/children"):
            return _FakeResponse(attachment_payload)
        if "/better-bibtex/json-rpc" in request.full_url:
            rpc_payload = json.loads(request.data.decode("utf-8"))
            rpc_payloads.append(rpc_payload)
            return _FakeResponse(
                {
                    "jsonrpc": "2.0",
                    "result": {"PAPER1": "yugayGaussianSLAMPhotorealisticDense"},
                    "id": rpc_payload.get("id"),
                }
            )
        return _FakeResponse(local_api_payload)

    with patch("citation_agent.zotero.urllib.request.urlopen", _urlopen):
        suggestions = ZoteroClient("http://127.0.0.1:23119/api").search(
            "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            limit=5,
            query_phrases=["Gaussian SLAM", "multi-agent Gaussian SLAM"],
            metadata_limit=10,
        )

    query_calls = [call for call in calls if "/items/top?" in call]
    assert len(query_calls) == 2
    assert "q=Gaussian+SLAM" in query_calls[0]
    assert rpc_payloads[0]["params"] == [["PAPER1"]]
    assert len(suggestions) == 1
    assert suggestions[0].citation_key == "yugayGaussianSLAMPhotorealisticDense"
    assert suggestions[0].pdf_path == "/mnt/c/Users/tester/Zotero/storage/PDF1/paper.pdf"


def test_recommender_prefers_evidence_supported_zotero_candidate(tmp_path):
    payload = [
        {
            "key": "SINGLE",
            "data": {
                "key": "SINGLE",
                "itemType": "journalArticle",
                "title": "Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting",
                "date": "2024",
                "citationKey": "yugayGaussianSLAMPhotorealisticDense",
                "abstractNote": "This paper studies dense SLAM with Gaussian splatting.",
            },
        },
        {
            "key": "MAGIC",
            "data": {
                "key": "MAGIC",
                "itemType": "conferencePaper",
                "title": "MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM",
                "date": "2025",
                "citationKey": "yugayMAGiCSLAMMultiAgentGaussian2025",
                "abstractNote": (
                    "This work studies multi-agent Gaussian SLAM. It uses offline "
                    "global optimization to maintain global consistency across agents."
                ),
            },
        },
    ]
    urlopen, _calls, _rpc_payloads = _make_urlopen(
        payload=payload,
        bbt_result={
            "SINGLE": "yugayGaussianSLAMPhotorealisticDense",
            "MAGIC": "yugayMAGiCSLAMMultiAgentGaussian2025",
        },
    )

    with patch("citation_agent.zotero.urllib.request.urlopen", urlopen):
        recommender = CitationRecommender(
            SourceConfig(
                max_suggestions=2,
                zotero_enabled=True,
                zotero_endpoint="http://127.0.0.1:23119/api",
                max_metadata_candidates=10,
                max_abstract_candidates=2,
                max_evidence_passages=3,
            )
        )
        suggestions = recommender.recommend(
            {
                "workspace_path": str(tmp_path),
                "file_path": str(tmp_path / "paper.tex"),
                "claim": "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
            }
        )

    assert suggestions[0].citation_key == "yugayMAGiCSLAMMultiAgentGaussian2025"
    assert suggestions[0].verdict == "direct_support"
    assert suggestions[1].verdict in {"related_only", "insufficient_evidence"}


def test_verify_evidence_limits_external_llm_judges(monkeypatch):
    monkeypatch.setenv("CITATION_AGENT_USE_LLM", "true")
    monkeypatch.setenv("CITATION_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("CITATION_AGENT_EVIDENCE_MAX_LLM_CANDIDATES", "1")
    monkeypatch.setenv("CITATION_AGENT_EVIDENCE_TIMEOUT_SECONDS", "8")
    calls: list[str] = []

    def fake_batch_llm_judge(self, claim, profile, items):  # noqa: ARG001
        judged = {}
        for suggestion, passages in items:
            calls.append(suggestion.title)
            judged[f"citekey:{suggestion.citation_key.lower()}"] = suggestion.model_copy(
                update={
                    "verdict": "partial_support",
                    "confidence": 0.7,
                    "evidence": passages[0],
                    "evidence_passages": passages,
                    "reason": "fake batch llm judge",
                }
            )
        return judged

    monkeypatch.setattr(EvidenceJudge, "_judge_batch_with_litellm", fake_batch_llm_judge)
    claim = "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization."
    profile = build_claim_profile(claim)
    suggestions = [
        CitationSuggestion(
            paper_id=f"paper:{index}",
            title=f"Candidate {index}",
            citation_key=f"candidate{index}",
            confidence=0.5,
            source="zotero",
            reason="test",
            evidence=EvidenceSnippet(
                text=(
                    "This work studies multi-agent Gaussian SLAM and offline global "
                    "optimization for globally consistent mapping."
                ),
                source="abstract",
                section="Abstract",
            ),
        )
        for index in range(3)
    ]

    verified = verify_evidence(
        claim=claim,
        profile=profile,
        suggestions=suggestions,
        max_passages=2,
    )

    assert calls == ["Candidate 0"]
    assert verified[0].reason == "fake batch llm judge"
    assert all(suggestion.reason != "fake batch llm judge" for suggestion in verified[1:])
    assert verify_evidence.last_stats["remote_candidates"] == 1
    assert verify_evidence.last_stats["cache_hits"] == 0
    assert verify_evidence.last_stats["local_fallback_candidates"] == 2
    assert verify_evidence.last_stats["skipped_by_limit"] == 2
