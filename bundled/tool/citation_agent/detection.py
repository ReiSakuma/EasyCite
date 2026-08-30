"""Citation need detection for selected LaTeX claims."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import ModelConfig
from .schemas import AuditClaimResult, CitationDetectionResult
from .tools.adapters import execute_litellm_completion


GENERALIZATION_PATTERNS = [
    r"\b(existing|prior|previous|recent|many|most|several|various|state-of-the-art)\b",
    r"\b(generally|typically|usually|commonly|often|widely|primarily)\b",
    r"\b(rely on|depend on|are based on|focus on|assume|require)\b",
]

EMPIRICAL_PATTERNS = [
    r"\b(outperform|improve|reduce|increase|achieve|show|demonstrate)\b",
    r"\b\d+(?:\.\d+)?\s?(%|percent|x|times)\b",
]

DEFINITION_PATTERNS = [
    r"\bis defined as\b",
    r"\brefers to\b",
    r"\bis known as\b",
]


class CitationNeedDetector:
    """Detect whether a claim needs citation support."""

    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig.from_env()

    def detect(self, params: dict[str, Any]) -> AuditClaimResult:
        claim = str(params.get("claim", "")).strip()
        context = str(params.get("context", "")).strip()
        existing_citations = params.get("existing_citations") or []

        if not claim:
            raise ValueError("audit_claim requires a non-empty claim.")

        if existing_citations:
            return AuditClaimResult(
                citation_needed=False,
                reason="选中文本中已经包含引用；V1 只判断是否缺少引用，暂不验证已有引用是否真实支持该陈述。",
                confidence=0.72,
                claim_type="other",
                source="heuristic",
            )

        if self.model_config.use_llm and self.model_config.api_key:
            try:
                return self._detect_with_litellm(claim=claim, context=context)
            except Exception as exc:  # pylint: disable=broad-except
                heuristic = self._detect_with_rules(claim)
                return heuristic.model_copy(
                    update={
                        "reason": f"{heuristic.reason}（LLM 检测失败，已回退本地规则：{type(exc).__name__}）",
                    }
                )

        return self._detect_with_rules(claim)

    def _detect_with_rules(self, claim: str) -> AuditClaimResult:
        lowered = claim.lower()

        if self._matches_any(lowered, EMPIRICAL_PATTERNS):
            return AuditClaimResult(
                citation_needed=True,
                reason="该句包含实验效果、数值结果或性能比较，通常需要引用或实验结果支撑。",
                confidence=0.86,
                claim_type="empirical_result",
                source="heuristic",
            )

        if self._matches_any(lowered, GENERALIZATION_PATTERNS):
            return AuditClaimResult(
                citation_needed=True,
                reason="该句概括了已有研究或领域常见做法，通常需要外部文献支持。",
                confidence=0.82,
                claim_type="related_work_summary",
                source="heuristic",
            )

        if self._matches_any(lowered, DEFINITION_PATTERNS):
            return AuditClaimResult(
                citation_needed=True,
                reason="该句给出术语定义或概念界定，若不是本文自定义概念，通常需要来源。",
                confidence=0.74,
                claim_type="definition",
                source="heuristic",
            )

        if len(claim) < 24:
            return AuditClaimResult(
                citation_needed=False,
                reason="选中文本较短，V1 本地规则未识别出明确需要引用的科研陈述。",
                confidence=0.55,
                claim_type="writing_only",
                source="heuristic",
            )

        return AuditClaimResult(
            citation_needed=False,
            reason="V1 本地规则未识别出领域概括、定义、实验结果或强方法断言。",
            confidence=0.58,
            claim_type="other",
            source="heuristic",
        )

    def _detect_with_litellm(self, claim: str, context: str) -> AuditClaimResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a citation-need detector for LaTeX scientific writing. "
                    "Return only valid JSON matching this schema: "
                    "{citation_needed: boolean, reason: string, confidence: number between 0 and 1, "
                    "claim_type: one of background, related_work_summary, method_claim, empirical_result, "
                    "definition, common_knowledge, writing_only, other}. "
                    "Do not invent papers or bibliography metadata."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Claim:\n{claim}\n\n"
                    f"Paragraph context:\n{context or '(none)'}\n\n"
                    "Judge whether this selected claim needs a citation."
                ),
            },
        ]

        kwargs: dict[str, Any] = {
            "model": self.model_config.model,
            "messages": messages,
            "api_key": self.model_config.api_key,
            "timeout": self.model_config.timeout_seconds,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self.model_config.api_base:
            kwargs["api_base"] = self.model_config.api_base

        response = execute_litellm_completion(kwargs)
        content = response.choices[0].message.content
        data = self._parse_json_object(content)
        detected = CitationDetectionResult(**data, source="llm")
        return AuditClaimResult(**detected.model_dump())

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))
