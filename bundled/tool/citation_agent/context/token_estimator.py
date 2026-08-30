"""Dependency-free, deterministic token estimation."""
from __future__ import annotations

import math
import re


def estimate_text_tokens(text: str) -> int:
    """Estimate model tokens without requiring a provider-specific tokenizer.

    ASCII prose averages close to four characters per token. CJK characters
    are conservatively counted as one token each. The estimate is intentionally
    stable across machines; provider usage can later be compared against it.
    """

    if not text:
        return 0
    cjk_count = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
    remaining = max(len(text) - cjk_count, 0)
    return max(1, math.ceil(remaining / 4.0) + cjk_count)


def truncate_text_to_tokens(text: str, target_tokens: int) -> str:
    """Deterministically shorten text while keeping a visible audit marker."""

    normalized = str(text or "")
    if target_tokens <= 0:
        return ""
    if estimate_text_tokens(normalized) <= target_tokens:
        return normalized

    suffix = " [truncated]"
    if estimate_text_tokens(suffix) >= target_tokens:
        return normalized[: max(target_tokens, 1)]

    low = 1
    high = len(normalized)
    while low < high:
        midpoint = (low + high + 1) // 2
        candidate = normalized[:midpoint].rstrip() + suffix
        if estimate_text_tokens(candidate) <= target_tokens:
            low = midpoint
        else:
            high = midpoint - 1
    prefix = normalized[:low].rstrip()
    boundary = max(prefix.rfind(" "), prefix.rfind("\n"))
    if boundary >= len(prefix) // 2:
        prefix = prefix[:boundary].rstrip()
    return prefix + suffix
