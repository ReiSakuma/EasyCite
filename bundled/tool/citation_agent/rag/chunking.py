"""Section-aware parent/child chunking for academic PDFs."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ..cache import stable_text_hash
from ..config import RagConfig
from ..context.token_estimator import estimate_text_tokens
from ..evidence import analyze_evidence_quality
from .models import RagChunk, RagDocument


@dataclass(frozen=True)
class TextUnit:
    text: str
    page: int | None
    section: str


SECTION_NAMES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methodology",
    "approach",
    "experiments",
    "experimental results",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "appendix",
    "references",
    "bibliography",
}


def chunk_document(document: RagDocument, config: RagConfig) -> list[RagChunk]:
    units: list[TextUnit] = []
    if document.abstract:
        units.append(TextUnit(normalize_text(document.abstract), None, "Abstract"))
    if document.pdf_path and os.path.isfile(document.pdf_path):
        units.extend(extract_pdf_units(document.pdf_path))
    return build_parent_child_chunks(document, units, config)


def extract_pdf_units(pdf_path: str) -> list[TextUnit]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return []

    units: list[TextUnit] = []
    section = "PDF"
    try:
        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text")
                paragraphs = split_page_paragraphs(text)
                for paragraph in paragraphs:
                    heading = detect_heading(paragraph)
                    if heading:
                        section = heading
                        if normalize_heading(heading) in {"references", "bibliography"}:
                            return units
                        continue
                    units.append(TextUnit(paragraph, page_index + 1, section))
    except Exception:  # pylint: disable=broad-except
        return units
    return units


def split_page_paragraphs(text: str) -> list[str]:
    normalized = str(text or "").replace("\r", "\n")
    normalized = re.sub(r"(?<=\w)-\n(?=[a-z])", "", normalized)
    blocks = re.split(r"\n\s*\n+", normalized)
    if len(blocks) <= 1:
        blocks = re.split(r"(?<=[.!?])\s*\n(?=[A-Z0-9])", normalized)
    paragraphs: list[str] = []
    for block in blocks:
        value = normalize_text(block)
        if value:
            paragraphs.append(value)
    return paragraphs


def detect_heading(text: str) -> str | None:
    value = normalize_text(text).strip(" .:")
    if not value or len(value) > 100 or len(value.split()) > 14:
        return None
    normalized = normalize_heading(value)
    without_number = re.sub(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*\.?\s+", "", normalized)
    if without_number in SECTION_NAMES:
        return value
    if re.match(r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)\s+[A-Z][A-Za-z]", value):
        return value
    letters = [char for char in value if char.isalpha()]
    if len(letters) >= 4 and value.upper() == value:
        return value.title()
    return None


def build_parent_child_chunks(
    document: RagDocument,
    units: list[TextUnit],
    config: RagConfig,
) -> list[RagChunk]:
    grouped: dict[str, list[TextUnit]] = {}
    order: list[str] = []
    for unit in units:
        if not unit.text:
            continue
        if unit.section not in grouped:
            grouped[unit.section] = []
            order.append(unit.section)
        grouped[unit.section].append(unit)

    chunks: list[RagChunk] = []
    ordinal = 0
    for section in order:
        section_units = grouped[section]
        sentence_units = [
            TextUnit(sentence, unit.page, section)
            for unit in section_units
            for sentence in split_sentences(unit.text)
        ]
        parents = make_windows(sentence_units, config.parent_chunk_tokens, 0)
        for parent_index, parent_units in enumerate(parents):
            parent_text = " ".join(unit.text for unit in parent_units).strip()
            if not parent_text:
                continue
            parent_id = stable_text_hash(
                f"{document.document_id}|{section}|{parent_index}|{parent_text}"
            )[:32]
            children = make_windows(
                parent_units,
                config.child_chunk_tokens,
                config.chunk_overlap_tokens,
                max_tokens=config.child_chunk_max_tokens,
            )
            for child_units in children:
                child_text = " ".join(unit.text for unit in child_units).strip()
                quality = analyze_evidence_quality(
                    child_text,
                    source="abstract" if section == "Abstract" else "pdf",
                )
                if quality.discard:
                    continue
                pages = [unit.page for unit in child_units if unit.page is not None]
                chunk_id = stable_text_hash(
                    f"{document.document_id}|{section}|{ordinal}|{child_text}"
                )
                chunks.append(
                    RagChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        parent_id=parent_id,
                        ordinal=ordinal,
                        section=section,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        text=child_text,
                        parent_text=parent_text,
                        chunk_type="abstract" if section == "Abstract" else "body",
                        quality_score=quality.quality_score,
                        quality_flags=tuple(quality.flags),
                    )
                )
                ordinal += 1
    return chunks


def make_windows(
    units: list[TextUnit],
    target_tokens: int,
    overlap_tokens: int,
    *,
    max_tokens: int | None = None,
) -> list[list[TextUnit]]:
    if not units:
        return []
    target = max(target_tokens, 32)
    maximum = max(max_tokens or target, target)
    windows: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    current_tokens = 0
    for unit in units:
        token_count = estimate_text_tokens(unit.text)
        if current and current_tokens + token_count > maximum:
            windows.append(current)
            current = overlap_tail(current, overlap_tokens)
            current_tokens = sum(estimate_text_tokens(item.text) for item in current)
        current.append(unit)
        current_tokens += token_count
        if current_tokens >= target:
            windows.append(current)
            current = overlap_tail(current, overlap_tokens)
            current_tokens = sum(estimate_text_tokens(item.text) for item in current)
    if current and (not windows or current != windows[-1]):
        windows.append(current)
    return dedupe_windows(windows)


def overlap_tail(units: list[TextUnit], overlap_tokens: int) -> list[TextUnit]:
    if overlap_tokens <= 0:
        return []
    kept: list[TextUnit] = []
    total = 0
    for unit in reversed(units):
        kept.append(unit)
        total += estimate_text_tokens(unit.text)
        if total >= overlap_tokens:
            break
    return list(reversed(kept))


def dedupe_windows(windows: list[list[TextUnit]]) -> list[list[TextUnit]]:
    seen: set[str] = set()
    result: list[list[TextUnit]] = []
    for window in windows:
        key = stable_text_hash(" ".join(unit.text for unit in window))
        if key in seen:
            continue
        seen.add(key)
        result.append(window)
    return result


def split_sentences(text: str) -> list[str]:
    value = normalize_text(text)
    if not value:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+(?=[A-Z0-9\u3400-\u9fff])", value)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip(":"))
