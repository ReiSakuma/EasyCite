"""On-demand bilingual evidence translation."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .config import ModelConfig
from .context import ContextItem, ContextManager, policy_for_stage
from .tools.adapters import execute_litellm_completion

TRANSLATION_BATCH_SIZE = 2
TRANSLATION_CHUNK_MAX_CHARS = 900


def translate_evidence(
    params: dict[str, Any],
    model_config: ModelConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Translate evidence texts into Simplified Chinese using the configured LLM."""

    config = model_config or ModelConfig.from_env()
    texts = normalize_translation_texts(params.get("texts"))
    if logger:
        logger(
            "Evidence translation request normalized: "
            f"text_count={len(texts)}, use_llm={config.use_llm}, "
            f"api_key={'set' if config.api_key else 'missing'}, model={config.model}."
        )
    if not texts:
        return {
            "available": False,
            "reason": "没有收到需要翻译的 evidence 文本。",
            "translations": [],
        }

    if not config.use_llm or not config.api_key:
        return {
            "available": False,
            "reason": "未配置模型 API，暂不生成中文对照。",
            "translations": [],
        }

    result, failures, context_reports = translate_with_litellm(texts=texts, config=config, logger=logger)
    if logger:
        for report in context_reports:
            logger(render_context_report(report))
    if result:
        if logger:
            logger(
                f"Translated {len(result)} evidence text(s) into Chinese; "
                f"failures={len(failures)}."
            )
        reason = "中文对照由已配置模型按需生成。"
        if failures:
            reason = (
                f"中文对照部分生成：成功 {len(result)} 段，失败 {len(failures)} 段。"
                f"失败段落：{', '.join(failures[:5])}。"
            )
        return {
            "available": True,
            "reason": reason,
            "translations": result,
            "context_reports": context_reports,
        }

    failure_reason = (
        f"中文对照生成失败：所有 {len(texts)} 段均未成功。"
        f"失败段落：{', '.join(failures[:5])}。"
        if failures
        else "模型没有返回可用中文翻译。"
    )
    return {
        "available": False,
        "reason": failure_reason,
        "translations": [],
        "context_reports": context_reports,
    }


def normalize_translation_texts(raw_texts: Any) -> list[dict[str, str]]:
    if not isinstance(raw_texts, list):
        return []

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_texts):
        text_id = str(get_field(item, "id") or f"text-{index}")
        text = re.sub(r"\s+", " ", str(get_field(item, "text") or "")).strip()
        if not text or text_id in seen:
            continue
        seen.add(text_id)
        normalized.append({"id": text_id, "text": text[:3000]})
    return normalized[:12]


def get_field(item: Any, field: str) -> Any:
    """Read a field from dict-like or pygls object-like request items."""

    if isinstance(item, Mapping):
        return item.get(field)

    if hasattr(item, field):
        return getattr(item, field)

    if hasattr(item, "__dict__"):
        return vars(item).get(field)

    return None


def translate_with_litellm(
    texts: list[dict[str, str]],
    config: ModelConfig,
    logger: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, str]], list[str], list[dict[str, Any]]]:
    translations: list[dict[str, str]] = []
    failures: list[str] = []
    context_reports: list[dict[str, Any]] = []

    chunk_items, chunk_groups = split_translation_texts(texts)
    for batch in chunked(chunk_items, TRANSLATION_BATCH_SIZE):
        try:
            setattr(call_litellm_translation, "last_context_report", None)
            translations.extend(call_litellm_translation(batch, config))
            capture_context_report(call_litellm_translation, context_reports)
            continue
        except Exception as exc:  # pylint: disable=broad-except
            capture_context_report(call_litellm_translation, context_reports)
            if logger:
                logger(
                    "Evidence translation batch failed; "
                    f"size={len(batch)}, ids={[item['id'] for item in batch]}, "
                    f"error={type(exc).__name__}: {exc}"
                )
            if len(batch) == 1:
                failures.append(f"{batch[0]['id']}({type(exc).__name__})")
                continue

        for item in batch:
            try:
                setattr(call_litellm_translation, "last_context_report", None)
                translations.extend(call_litellm_translation([item], config))
                capture_context_report(call_litellm_translation, context_reports)
            except Exception as exc:  # pylint: disable=broad-except
                capture_context_report(call_litellm_translation, context_reports)
                failures.append(f"{item['id']}({type(exc).__name__})")

    combined, missing = combine_translation_chunks(translations, chunk_groups)
    failures.extend(missing)
    return dedupe_translations(combined), failures, context_reports


def call_litellm_translation(
    texts: list[dict[str, str]],
    config: ModelConfig,
) -> list[dict[str, str]]:
    system_prompt = (
        "You translate scientific evidence into Simplified Chinese. "
        "Preserve technical terms, citation markers like [12], arXiv IDs, math, LaTeX, "
        "proper nouns, and paper titles when a literal translation would be confusing. "
        "Return only valid JSON: {\"translations\":[{\"id\":\"...\",\"zh\":\"...\"}]}."
    )
    context_packet = ContextManager(
        policy_for_stage(
            "translation",
            translation_budget=config.context_translation_budget_tokens,
        )
    ).build_json_messages(
        system_prompt=system_prompt,
        items=[
            ContextItem(
                "task",
                "Translate each text into accurate academic Simplified Chinese.",
                priority="required",
                source="translation_node_policy",
                purpose="define_translation_task",
            ),
            ContextItem(
                "texts",
                texts,
                priority="compressible",
                source="final_candidate_evidence",
                purpose="produce_bilingual_user_review_text",
                min_tokens=600,
                max_tokens=max(config.context_translation_budget_tokens - 500, 800),
                min_items=len(texts),
                protected_keys=("id",),
            ),
        ],
    )
    setattr(call_litellm_translation, "last_context_report", context_packet.report.to_dict())
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": context_packet.messages,
        "api_key": config.api_key,
        "timeout": max(config.timeout_seconds, config.translation_timeout_seconds),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if config.api_base:
        kwargs["api_base"] = config.api_base

    response = execute_litellm_completion(kwargs)
    data = parse_json_object(response.choices[0].message.content)
    translations = data.get("translations")
    if not isinstance(translations, list):
        return []

    usable: list[dict[str, str]] = []
    allowed_ids = {item["id"] for item in texts}
    for item in translations:
        if not isinstance(item, dict):
            continue
        text_id = str(item.get("id") or "")
        zh = str(item.get("zh") or "").strip()
        if text_id in allowed_ids and zh:
            usable.append({"id": text_id, "zh": zh})
    return usable


def capture_context_report(callable_obj: Any, reports: list[dict[str, Any]]) -> None:
    report = getattr(callable_obj, "last_context_report", None)
    if isinstance(report, dict):
        reports.append(report)


def render_context_report(report: dict[str, Any]) -> str:
    return (
        f"Context budget stage={report.get('stage', 'translation')}, "
        f"estimated_tokens={report.get('original_tokens', 0)}->"
        f"{report.get('final_tokens', 0)}/{report.get('budget_tokens', 0)}, "
        f"budget_exceeded={bool(report.get('budget_exceeded'))}."
    )


def chunked(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def split_translation_texts(
    texts: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    chunks: list[dict[str, str]] = []
    groups: dict[str, list[str]] = {}
    for item in texts:
        source_id = item["id"]
        text_chunks = split_text_for_translation(item["text"])
        groups[source_id] = []
        for index, chunk in enumerate(text_chunks):
            chunk_id = source_id if len(text_chunks) == 1 else f"{source_id}__part{index + 1}"
            groups[source_id].append(chunk_id)
            chunks.append({"id": chunk_id, "text": chunk})
    return chunks, groups


def split_text_for_translation(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= TRANSLATION_CHUNK_MAX_CHARS:
        return [normalized] if normalized else []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= TRANSLATION_CHUNK_MAX_CHARS:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        while len(sentence) > TRANSLATION_CHUNK_MAX_CHARS:
            chunks.append(sentence[:TRANSLATION_CHUNK_MAX_CHARS].strip())
            sentence = sentence[TRANSLATION_CHUNK_MAX_CHARS:].strip()
        buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks


def combine_translation_chunks(
    translations: list[dict[str, str]],
    chunk_groups: dict[str, list[str]],
) -> tuple[list[dict[str, str]], list[str]]:
    by_id = {item["id"]: item["zh"] for item in translations if item.get("id") and item.get("zh")}
    combined: list[dict[str, str]] = []
    failures: list[str] = []
    for source_id, chunk_ids in chunk_groups.items():
        translated_chunks = [by_id.get(chunk_id) for chunk_id in chunk_ids]
        if all(translated_chunks):
            combined.append({"id": source_id, "zh": "\n\n".join(str(chunk) for chunk in translated_chunks)})
            continue
        missing = [chunk_id for chunk_id, translated in zip(chunk_ids, translated_chunks) if not translated]
        failures.append(f"{source_id}({len(missing)}/{len(chunk_ids)} chunks failed)")
        if any(translated_chunks):
            partial = [str(chunk) for chunk in translated_chunks if chunk]
            partial.append("（该段中文对照部分生成，仍有小块翻译失败。）")
            combined.append({"id": source_id, "zh": "\n\n".join(partial)})
    return combined, failures


def dedupe_translations(translations: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in translations:
        text_id = item.get("id")
        if not text_id or text_id in seen:
            continue
        seen.add(text_id)
        deduped.append(item)
    return deduped


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
