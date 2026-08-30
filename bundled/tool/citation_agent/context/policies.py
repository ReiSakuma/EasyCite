"""Node-level token budgets for EasyCite model calls."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPolicy:
    stage: str
    token_budget: int


def policy_for_stage(
    stage: str,
    *,
    query_budget: int = 3200,
    evidence_budget: int = 9000,
    translation_budget: int = 5000,
) -> ContextPolicy:
    budgets = {
        "query_planning": query_budget,
        "evidence_verification": evidence_budget,
        "translation": translation_budget,
    }
    return ContextPolicy(stage=stage, token_budget=max(int(budgets.get(stage, 4000)), 256))
