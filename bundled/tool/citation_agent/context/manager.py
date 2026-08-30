"""Build deterministic, budgeted JSON context for LLM calls."""
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from .models import BudgetReport, ContextItem, ContextItemReport, ContextPacket
from .policies import ContextPolicy
from .token_estimator import estimate_text_tokens, truncate_text_to_tokens


class ContextManager:
    """Apply per-item limits and a total node budget without changing schemas."""

    def __init__(
        self,
        policy: ContextPolicy,
        logger: Callable[[str], None] | None = None,
    ):
        self.policy = policy
        self.logger = logger

    def build_json_messages(
        self,
        *,
        system_prompt: str,
        items: list[ContextItem],
    ) -> ContextPacket:
        system_tokens = estimate_text_tokens(system_prompt)
        working: list[dict[str, Any]] = []
        original_item_tokens = {
            item.name: estimate_value_tokens(item.value)
            for item in items
        }

        for item in items:
            original_value = copy.deepcopy(item.value)
            value = dedupe_value(original_value)
            deduplicated = value != original_value
            if item.max_tokens is not None and estimate_value_tokens(value) > item.max_tokens:
                value = compact_value(
                    value,
                    target_tokens=max(item.max_tokens, item.min_tokens),
                    min_items=item.min_items,
                    protected_keys=set(item.protected_keys),
                )
            working.append(
                {
                    "item": item,
                    "value": value,
                    "dropped": False,
                    "deduplicated": deduplicated,
                }
            )

        original_payload = {item.name: item.value for item in items}
        original_tokens = system_tokens + estimate_text_tokens(
            json.dumps(original_payload, ensure_ascii=False, separators=(",", ":"))
        )
        self._drop_optional_items(working, system_tokens)
        self._compact_to_total_budget(working, system_tokens)

        payload = {
            entry["item"].name: entry["value"]
            for entry in working
            if not entry["dropped"]
        }
        user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        final_tokens = system_tokens + estimate_text_tokens(user_content)
        reports: list[ContextItemReport] = []
        for entry in working:
            item = entry["item"]
            original = original_item_tokens[item.name]
            final = 0 if entry["dropped"] else estimate_value_tokens(entry["value"])
            action = (
                "dropped"
                if entry["dropped"]
                else ("compacted" if final < original or entry["deduplicated"] else "kept")
            )
            if entry["dropped"]:
                reason = "droppable_item_removed_to_fit_node_budget"
            elif action == "compacted":
                reason = "deduplicated_or_compacted_to_fit_item_or_node_budget"
            elif item.priority == "required" and final_tokens > self.policy.token_budget:
                reason = "required_item_preserved_despite_budget_overflow"
            else:
                reason = "within_budget"
            reports.append(
                ContextItemReport(
                    name=item.name,
                    priority=item.priority,
                    source=item.source,
                    purpose=item.purpose,
                    original_tokens=original,
                    final_tokens=final,
                    action=action,
                    reason=reason,
                )
            )

        report = BudgetReport(
            stage=self.policy.stage,
            budget_tokens=self.policy.token_budget,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            system_tokens=system_tokens,
            budget_exceeded=final_tokens > self.policy.token_budget,
            items=tuple(reports),
        )
        if self.logger:
            self.logger(report.summary())
        return ContextPacket(
            stage=self.policy.stage,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            payload=payload,
            report=report,
        )

    def _drop_optional_items(self, working: list[dict[str, Any]], system_tokens: int) -> None:
        for entry in reversed(working):
            if current_total_tokens(working, system_tokens) <= self.policy.token_budget:
                return
            item: ContextItem = entry["item"]
            if item.priority == "droppable":
                entry["dropped"] = True

    def _compact_to_total_budget(self, working: list[dict[str, Any]], system_tokens: int) -> None:
        for entry in reversed(working):
            current_total = current_total_tokens(working, system_tokens)
            if current_total <= self.policy.token_budget:
                return
            item: ContextItem = entry["item"]
            if entry["dropped"] or item.priority != "compressible":
                continue
            current = estimate_value_tokens(entry["value"])
            excess = current_total - self.policy.token_budget
            target = max(item.min_tokens, current - excess)
            entry["value"] = compact_value(
                entry["value"],
                target_tokens=target,
                min_items=item.min_items,
                protected_keys=set(item.protected_keys),
            )


def current_total_tokens(working: list[dict[str, Any]], system_tokens: int) -> int:
    payload = {
        entry["item"].name: entry["value"]
        for entry in working
        if not entry["dropped"]
    }
    return system_tokens + estimate_text_tokens(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def estimate_value_tokens(value: Any) -> int:
    return estimate_text_tokens(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def dedupe_value(value: Any) -> Any:
    """Remove exact repeated list entries while preserving first-seen order."""

    if isinstance(value, dict):
        return {key: dedupe_value(item) for key, item in value.items()}
    if not isinstance(value, list):
        return value

    result: list[Any] = []
    seen: set[str] = set()
    for item in value:
        compacted_item = dedupe_value(item)
        try:
            identity = json.dumps(
                compacted_item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            identity = repr(compacted_item)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(compacted_item)
    return result


def compact_value(
    value: Any,
    *,
    target_tokens: int,
    min_items: int = 0,
    protected_keys: set[str] | None = None,
) -> Any:
    """Shrink string leaves first, preserving dictionaries and required list items."""

    protected = protected_keys or set()
    compacted = copy.deepcopy(value)
    target = max(int(target_tokens), 0)
    if estimate_value_tokens(compacted) <= target:
        return compacted
    if isinstance(compacted, str):
        return truncate_text_to_tokens(compacted, target)

    for _ in range(64):
        current = estimate_value_tokens(compacted)
        if current <= target:
            break
        leaves = collect_string_leaves(compacted, protected)
        if leaves:
            path, text = max(leaves, key=lambda entry: (estimate_text_tokens(entry[1]), repr(entry[0])))
            current_text_tokens = estimate_text_tokens(text)
            excess = current - target
            next_target = max(8, current_text_tokens - max(excess, max(current_text_tokens // 4, 1)))
            replacement = truncate_text_to_tokens(text, next_target)
            if replacement == text:
                break
            set_path_value(compacted, path, replacement)
            continue
        if isinstance(compacted, list) and len(compacted) > min_items:
            compacted.pop()
            continue
        break
    return compacted


def collect_string_leaves(value: Any, protected_keys: set[str], path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], str]]:
    leaves: list[tuple[tuple[Any, ...], str]] = []
    if isinstance(value, str):
        if not path or str(path[-1]) not in protected_keys:
            leaves.append((path, value))
        return leaves
    if isinstance(value, list):
        for index, item in enumerate(value):
            leaves.extend(collect_string_leaves(item, protected_keys, (*path, index)))
    elif isinstance(value, dict):
        for key in sorted(value):
            leaves.extend(collect_string_leaves(value[key], protected_keys, (*path, key)))
    return leaves


def set_path_value(root: Any, path: tuple[Any, ...], value: Any) -> None:
    if not path:
        return
    current = root
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
