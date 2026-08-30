"""Policy deciding whether a node can be replayed, reused or restarted."""
from __future__ import annotations

from typing import Literal


RecoveryAction = Literal["replay", "reuse", "restart", "reject"]


def recovery_action(node: str, *, graph_compatible: bool, cancelled: bool = False) -> RecoveryAction:
    if not graph_compatible or cancelled:
        return "reject"
    if node in {"initialize", "validate_request", "analyze_claim", "plan_retrieval"}:
        return "replay"
    if node in {"recall_current_bib", "recall_zotero", "select_evidence_targets", "verify_evidence"}:
        return "reuse"
    if node in {"dedupe_candidates", "scope_filter", "metadata_rank", "rank_verified_evidence", "finalize_recommendations"}:
        return "replay"
    if node == "persist_run":
        return "restart"
    return "reject"
