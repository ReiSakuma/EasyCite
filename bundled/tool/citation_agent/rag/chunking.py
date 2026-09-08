"""Paragraph-aware semantic parent/child chunking for academic PDFs."""
from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from ..cache import stable_text_hash
from ..config import RagConfig
from ..context.token_estimator import estimate_text_tokens
from ..evidence import analyze_evidence_quality
from .models import RagChunk, RagDocument


ParagraphEmbedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class TextUnit:
    text: str
    page: int | None
    section: str
    paragraph_id: int = 0
    block_type: str = "body"


@dataclass(frozen=True)
class LayoutBlock:
    """A text block with enough page geometry to rebuild logical paragraphs."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    line_count: int

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 0.0)


@dataclass(frozen=True)
class SemanticParent:
    units: tuple[TextUnit, ...]
    boundary_reason: str
    boundary_score: float | None = None


@dataclass(frozen=True)
class ChunkingStats:
    paragraphs: int = 0
    semantic_embedding_inputs: int = 0
    parents: int = 0
    semantic_boundaries: int = 0
    adaptive_boundaries: int = 0
    size_boundaries: int = 0
    semantic_used: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ChunkingResult:
    chunks: tuple[RagChunk, ...]
    stats: ChunkingStats


SECTION_NAMES = {
    "abstract", "introduction", "background", "related work", "method",
    "methodology", "approach", "experiments", "experimental results", "results",
    "discussion", "conclusion", "conclusions", "limitations", "appendix",
    "references", "bibliography",
}


def chunk_document(document: RagDocument, config: RagConfig) -> list[RagChunk]:
    """Compatibility wrapper for callers that do not need semantic diagnostics."""
    return list(chunk_document_with_stats(document, config).chunks)


def chunk_document_with_stats(
    document: RagDocument,
    config: RagConfig,
    *,
    paragraph_embedder: ParagraphEmbedder | None = None,
) -> ChunkingResult:
    units: list[TextUnit] = []
    if document.abstract:
        units.append(TextUnit(
            normalize_text(document.abstract), None, "Abstract",
            paragraph_id=0, block_type="abstract",
        ))
    if document.pdf_path and os.path.isfile(document.pdf_path):
        units.extend(extract_pdf_units(document.pdf_path, paragraph_offset=len(units)))
    return build_parent_child_chunks_with_stats(
        document, units, config, paragraph_embedder=paragraph_embedder,
    )


def extract_pdf_units(pdf_path: str, *, paragraph_offset: int = 0) -> list[TextUnit]:
    try:
        try:
            import pymupdf as fitz  # type: ignore
        except ImportError:
            import fitz  # type: ignore
    except ImportError:
        return []

    units: list[TextUnit] = []
    section = "PDF"
    paragraph_id = max(paragraph_offset, 0)
    try:
        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                for paragraph in extract_page_paragraphs(page):
                    heading = detect_heading(paragraph)
                    if heading:
                        section = heading
                        if normalize_heading(heading) in {"references", "bibliography"}:
                            return units
                        continue
                    units.append(TextUnit(
                        paragraph, page_index + 1, section,
                        paragraph_id=paragraph_id, block_type="body",
                    ))
                    paragraph_id += 1
    except Exception:  # pylint: disable=broad-except
        return units
    return units


def extract_page_paragraphs(page) -> list[str]:
    """Rebuild logical paragraphs from layout blocks instead of indexing PDF lines."""
    try:
        blocks = page.get_text("blocks", sort=True)
        page_width = float(getattr(page.rect, "width", 0.0) or 0.0)
        page_height = float(getattr(page.rect, "height", 0.0) or 0.0)
        layout_blocks = parse_layout_blocks(blocks, page_width, page_height)
        paragraphs = merge_layout_blocks(layout_blocks, page_width)
    except Exception:  # pylint: disable=broad-except
        paragraphs = []
    return paragraphs or split_page_paragraphs(page.get_text("text"))


def parse_layout_blocks(
    blocks: list[tuple], page_width: float, page_height: float,
) -> list[LayoutBlock]:
    parsed: list[LayoutBlock] = []
    for block in blocks:
        if len(block) >= 7 and int(block[6]) != 0:
            continue
        raw_text = str(block[4]) if len(block) >= 5 else ""
        raw_lines = [line.strip() for line in raw_text.replace("\r", "\n").splitlines() if line.strip()]
        text = normalize_pdf_block(raw_text)
        if not text:
            continue
        layout = LayoutBlock(
            text=text,
            x0=float(block[0]),
            y0=float(block[1]),
            x1=float(block[2]),
            y1=float(block[3]),
            line_count=max(len(raw_lines), 1),
        )
        if is_layout_noise(layout, page_width, page_height):
            continue
        parsed.append(layout)
    return parsed


def merge_layout_blocks(blocks: list[LayoutBlock], page_width: float) -> list[str]:
    if not blocks:
        return []
    line_heights = [
        block.height / max(block.line_count, 1)
        for block in blocks
        if block.height > 0
    ]
    median_line_height = percentile(line_heights, 0.5) or 10.0
    dominant_left = dominant_left_edge(blocks)
    paragraphs: list[str] = []
    current = blocks[0].text
    previous = blocks[0]

    for block in blocks[1:]:
        if should_merge_layout_blocks(
            previous,
            block,
            page_width=page_width,
            median_line_height=median_line_height,
            dominant_left=dominant_left,
        ):
            current = join_pdf_fragments(current, block.text)
        else:
            if value := normalize_text(current):
                paragraphs.append(value)
            current = block.text
        previous = block

    if value := normalize_text(current):
        paragraphs.append(value)
    return paragraphs


def should_merge_layout_blocks(
    previous: LayoutBlock,
    current: LayoutBlock,
    *,
    page_width: float,
    median_line_height: float,
    dominant_left: float,
) -> bool:
    # Multi-line blocks from well-formed PDFs already represent paragraphs.
    if previous.line_count > 2 or current.line_count > 2:
        return False
    if len(previous.text.split()) > 55 or len(current.text.split()) > 55:
        return False
    if detect_heading(previous.text) or detect_heading(current.text):
        return False
    if not same_text_column(previous, current, page_width):
        return False
    vertical_gap = current.y0 - previous.y1
    if vertical_gap < -0.35 * median_line_height:
        return False
    if vertical_gap > max(0.85 * median_line_height, 7.0):
        return False
    indent_threshold = max(page_width * 0.012, 7.0)
    previous_is_list_item = bool(re.match(r"^(?:[•·▪◦*-]|\(?\d+[.)])\s*", previous.text))
    if current.x0 - previous.x0 >= indent_threshold and not previous_is_list_item:
        return False
    current_is_indented = current.x0 - dominant_left >= indent_threshold
    previous_is_indented = previous.x0 - dominant_left >= indent_threshold
    if current_is_indented and not previous_is_indented:
        return False
    return True


def same_text_column(first: LayoutBlock, second: LayoutBlock, page_width: float) -> bool:
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    smaller_width = max(min(first.width, second.width), 1.0)
    left_tolerance = max(page_width * 0.04, 18.0)
    return overlap / smaller_width >= 0.55 or abs(first.x0 - second.x0) <= left_tolerance


def dominant_left_edge(blocks: list[LayoutBlock]) -> float:
    candidates = [block.x0 for block in blocks if block.line_count <= 2 and len(block.text.split()) >= 5]
    if not candidates:
        candidates = [block.x0 for block in blocks]
    buckets: dict[int, list[float]] = {}
    for value in candidates:
        buckets.setdefault(round(value / 4), []).append(value)
    largest = max(buckets.values(), key=len)
    return sum(largest) / len(largest)


def is_layout_noise(block: LayoutBlock, page_width: float, page_height: float) -> bool:
    text = block.text.strip()
    words = text.split()
    normalized = normalize_heading(text)
    known_section = re.sub(
        r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*\.?\s+", "", normalized,
    ) in SECTION_NAMES
    if re.fullmatch(r"(?:page\s*)?\d{1,4}", text, flags=re.IGNORECASE):
        return True
    if page_height > 0 and len(words) <= 12 and not known_section:
        if block.y0 <= page_height * 0.045 or block.y0 >= page_height * 0.955:
            return True
    if page_height > 0 and page_width > 0:
        if block.height > page_height * 0.55 and block.width < page_width * 0.12:
            return True
    caption_prefix = re.sub(r"\s+", "", text[:24]).lower()
    if re.match(r"^(?:fig(?:ure)?|table)[a-z0-9ivx.-]+", caption_prefix):
        return True
    alphanumeric = sum(char.isalnum() for char in text)
    alphabetic = sum(char.isalpha() for char in text)
    if len(text) >= 6 and alphanumeric > 0 and alphabetic / alphanumeric < 0.35 and len(words) < 24:
        return True
    return False


def normalize_pdf_block(text: str) -> str:
    value = str(text or "").replace("\u00ad", "").replace("\r", "\n")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", value)
    return normalize_text(value)


def join_pdf_fragments(first: str, second: str) -> str:
    left = first.rstrip()
    right = second.lstrip()
    if re.search(r"[A-Za-z]-$", left) and re.match(r"^[a-z]", right):
        return f"{left[:-1]}{right}"
    return f"{left} {right}"


def split_page_paragraphs(text: str) -> list[str]:
    normalized = str(text or "").replace("\r", "\n")
    normalized = re.sub(r"(?<=\w)-\n(?=[a-z])", "", normalized)
    blocks = re.split(r"\n\s*\n+", normalized)
    if len(blocks) <= 1:
        blocks = re.split(r"(?<=[.!?])\s*\n(?=[A-Z0-9])", normalized)
    return [value for block in blocks if (value := normalize_text(block))]


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
    *,
    paragraph_embedder: ParagraphEmbedder | None = None,
) -> list[RagChunk]:
    return list(build_parent_child_chunks_with_stats(
        document, units, config, paragraph_embedder=paragraph_embedder,
    ).chunks)


def build_parent_child_chunks_with_stats(
    document: RagDocument,
    units: list[TextUnit],
    config: RagConfig,
    *,
    paragraph_embedder: ParagraphEmbedder | None = None,
) -> ChunkingResult:
    normalized_units = [unit for unit in units if normalize_text(unit.text)]
    chunks: list[RagChunk] = []
    ordinal = 0
    parent_count = semantic_boundaries = adaptive_boundaries = size_boundaries = 0
    semantic_used = False
    fallback_reasons: list[str] = []

    section_groups = [
        expand_oversized_paragraphs(group, config.parent_chunk_tokens)
        for group in contiguous_section_groups(normalized_units)
    ]
    all_expanded = [unit for group in section_groups for unit in group]
    all_vectors, embedding_fallback, semantic_embedding_inputs = embed_paragraph_units(
        all_expanded, config, paragraph_embedder,
    )
    vector_offset = 0

    for section_units in section_groups:
        section_vectors = (
            all_vectors[vector_offset : vector_offset + len(section_units)]
            if all_vectors else []
        )
        vector_offset += len(section_units)
        parents, section_stats = make_semantic_parents(
            section_units,
            config,
            paragraph_embedder=None,
            precomputed_vectors=section_vectors if all_vectors else None,
            units_are_expanded=True,
            inherited_fallback=embedding_fallback,
        )
        parent_count += len(parents)
        semantic_boundaries += section_stats.semantic_boundaries
        adaptive_boundaries += section_stats.adaptive_boundaries
        size_boundaries += section_stats.size_boundaries
        semantic_used = semantic_used or section_stats.semantic_used
        if section_stats.fallback_reason:
            fallback_reasons.append(section_stats.fallback_reason)

        for parent in parents:
            parent_units = [
                TextUnit(
                    sentence, unit.page, unit.section,
                    paragraph_id=unit.paragraph_id, block_type=unit.block_type,
                )
                for unit in parent.units
                for sentence in split_sentences(unit.text)
            ]
            parent_text = " ".join(unit.text for unit in parent_units).strip()
            if not parent_text:
                continue
            paragraph_ids = [unit.paragraph_id for unit in parent.units]
            paragraph_start = min(paragraph_ids) if paragraph_ids else None
            paragraph_end = max(paragraph_ids) if paragraph_ids else None
            section = parent.units[0].section
            parent_id = stable_text_hash(
                f"{document.document_id}|{section}|{paragraph_start}|{paragraph_end}|{parent_text}"
            )[:32]
            children = make_windows(
                parent_units,
                config.child_chunk_tokens,
                config.chunk_overlap_tokens,
                max_tokens=config.child_chunk_max_tokens,
            )
            for child_index, child_units in enumerate(children):
                child_text = " ".join(unit.text for unit in child_units).strip()
                source = "abstract" if child_units[0].block_type == "abstract" else "pdf"
                quality = analyze_evidence_quality(child_text, source=source)
                if quality.discard:
                    continue
                pages = [unit.page for unit in child_units if unit.page is not None]
                child_paragraphs = [unit.paragraph_id for unit in child_units]
                chunk_id = stable_text_hash(
                    f"{document.document_id}|{parent_id}|{child_index}|{child_text}"
                )
                chunks.append(RagChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    parent_id=parent_id,
                    ordinal=ordinal,
                    section=section,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    text=child_text,
                    parent_text=parent_text,
                    chunk_type="abstract" if source == "abstract" else "body",
                    quality_score=quality.quality_score,
                    quality_flags=tuple(quality.flags),
                    paragraph_start=min(child_paragraphs) if child_paragraphs else None,
                    paragraph_end=max(child_paragraphs) if child_paragraphs else None,
                    boundary_reason=parent.boundary_reason,
                    boundary_score=parent.boundary_score,
                ))
                ordinal += 1

    return ChunkingResult(
        chunks=tuple(chunks),
        stats=ChunkingStats(
            paragraphs=len(normalized_units),
            semantic_embedding_inputs=semantic_embedding_inputs,
            parents=parent_count,
            semantic_boundaries=semantic_boundaries,
            adaptive_boundaries=adaptive_boundaries,
            size_boundaries=size_boundaries,
            semantic_used=semantic_used,
            fallback_reason="; ".join(dict.fromkeys(fallback_reasons)) or None,
        ),
    )


def contiguous_section_groups(units: list[TextUnit]) -> list[list[TextUnit]]:
    groups: list[list[TextUnit]] = []
    for unit in units:
        if not groups or groups[-1][-1].section != unit.section:
            groups.append([unit])
        else:
            groups[-1].append(unit)
    return groups


def make_semantic_parents(
    units: list[TextUnit],
    config: RagConfig,
    *,
    paragraph_embedder: ParagraphEmbedder | None,
    precomputed_vectors: list[list[float]] | None = None,
    units_are_expanded: bool = False,
    inherited_fallback: str | None = None,
) -> tuple[list[SemanticParent], ChunkingStats]:
    expanded = units if units_are_expanded else expand_oversized_paragraphs(
        units, config.parent_chunk_tokens,
    )
    if not expanded:
        return [], ChunkingStats()

    vectors: list[list[float]] = []
    fallback_reason: str | None = inherited_fallback
    semantic_used = False
    if precomputed_vectors is not None:
        vectors = precomputed_vectors
        semantic_used = len(vectors) == len(expanded) and all(vectors)
        if not semantic_used:
            fallback_reason = "semantic chunking unavailable: incomplete paragraph vectors"
            vectors = []
    elif inherited_fallback is not None:
        vectors = []
    elif config.semantic_chunking_enabled and len(expanded) > 1:
        if paragraph_embedder is None:
            fallback_reason = "semantic chunking unavailable: embedding is disabled"
        else:
            try:
                vectors = paragraph_embedder([unit.text for unit in expanded])
                semantic_used = len(vectors) == len(expanded) and all(vectors)
                if not semantic_used:
                    fallback_reason = "semantic chunking unavailable: incomplete paragraph vectors"
                    vectors = []
            except Exception as exc:  # pylint: disable=broad-except
                fallback_reason = f"semantic chunking unavailable: {type(exc).__name__}: {exc}"
                vectors = []

    boundary_scores = calculate_boundary_scores(
        vectors, max(config.semantic_context_window, 1),
    ) if vectors else [None] * len(expanded)
    comparable_scores = [score for score in boundary_scores[1:] if score is not None]
    adaptive_threshold: float | None = None
    if config.semantic_adaptive_enabled and len(comparable_scores) >= 3:
        adaptive_threshold = max(
            config.semantic_boundary_threshold,
            min(
                percentile(comparable_scores, config.semantic_adaptive_percentile),
                config.semantic_adaptive_max_threshold,
            ),
        )

    parents: list[SemanticParent] = []
    current: list[TextUnit] = []
    current_tokens = 0
    boundary_reason = "section"
    boundary_score: float | None = None
    semantic_boundaries = adaptive_boundaries = size_boundaries = 0

    for index, unit in enumerate(expanded):
        unit_tokens = estimate_text_tokens(unit.text)
        reason: str | None = None
        score: float | None = None
        if current and vectors:
            score = boundary_scores[index]
            if score < config.semantic_hard_boundary_threshold:
                reason = "semantic_hard"
            elif (
                current_tokens >= max(config.semantic_min_parent_tokens, 0)
                and score < config.semantic_boundary_threshold
            ):
                reason = "semantic_soft"
            elif (
                current_tokens >= max(config.semantic_min_parent_tokens, 0)
                and adaptive_threshold is not None
                and score <= adaptive_threshold
                and is_local_semantic_low(
                    boundary_scores,
                    index,
                    min_drop=max(config.semantic_min_score_drop, 0.0),
                )
            ):
                reason = "semantic_adaptive"
        if (
            reason is None
            and current
            and current_tokens + unit_tokens > max(config.parent_chunk_tokens, 32)
        ):
            reason = "size_limit"

        if reason:
            parents.append(SemanticParent(tuple(current), boundary_reason, boundary_score))
            if reason == "size_limit":
                size_boundaries += 1
            else:
                semantic_boundaries += 1
                if reason == "semantic_adaptive":
                    adaptive_boundaries += 1
            current = []
            current_tokens = 0
            boundary_reason = reason
            boundary_score = round(score, 4) if score is not None else None

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        parents.append(SemanticParent(tuple(current), boundary_reason, boundary_score))

    return parents, ChunkingStats(
        paragraphs=len(expanded),
        parents=len(parents),
        semantic_boundaries=semantic_boundaries,
        adaptive_boundaries=adaptive_boundaries,
        size_boundaries=size_boundaries,
        semantic_used=semantic_used,
        fallback_reason=fallback_reason,
    )


def embed_paragraph_units(
    units: list[TextUnit],
    config: RagConfig,
    paragraph_embedder: ParagraphEmbedder | None,
) -> tuple[list[list[float]], str | None, int]:
    """Embed all paragraphs once so section count does not multiply model overhead."""
    if not config.semantic_chunking_enabled or len(units) <= 1:
        return [], None, 0
    if paragraph_embedder is None:
        return [], "semantic chunking unavailable: embedding is disabled", 0
    texts, unit_to_input = semantic_embedding_inputs(
        units, max(config.semantic_embedding_min_tokens, 0),
    )
    try:
        input_vectors = paragraph_embedder(texts)
        if len(input_vectors) != len(texts) or not all(input_vectors):
            return [], "semantic chunking unavailable: incomplete paragraph vectors", len(texts)
        vectors = [input_vectors[input_index] for input_index in unit_to_input]
        return vectors, None, len(texts)
    except Exception as exc:  # pylint: disable=broad-except
        return [], f"semantic chunking unavailable: {type(exc).__name__}: {exc}", len(texts)


def semantic_embedding_inputs(
    units: list[TextUnit], min_tokens: int,
) -> tuple[list[str], list[int]]:
    """Attach short layout fragments to prose anchors for cheaper, stabler semantics."""
    if min_tokens <= 0:
        return [unit.text for unit in units], list(range(len(units)))

    texts: list[str] = []
    unit_to_input = [-1] * len(units)
    start = 0
    while start < len(units):
        end = start + 1
        while end < len(units) and units[end].section == units[start].section:
            end += 1
        pending: list[int] = []
        section_inputs: list[int] = []
        for index in range(start, end):
            pending.append(index)
            if estimate_text_tokens(units[index].text) < min_tokens:
                continue
            input_index = len(texts)
            texts.append(" ".join(units[item].text for item in pending))
            for item in pending:
                unit_to_input[item] = input_index
            section_inputs.append(input_index)
            pending = []
        if pending:
            if section_inputs:
                input_index = section_inputs[-1]
                texts[input_index] = join_pdf_fragments(
                    texts[input_index], " ".join(units[item].text for item in pending),
                )
            else:
                input_index = len(texts)
                texts.append(" ".join(units[item].text for item in pending))
            for item in pending:
                unit_to_input[item] = input_index
        start = end
    return texts, unit_to_input


def calculate_boundary_scores(
    vectors: list[list[float]], context_window: int,
) -> list[float | None]:
    """Score each inter-paragraph boundary using adjacent and bilateral context."""
    scores: list[float | None] = [None]
    window = max(context_window, 1)
    for index in range(1, len(vectors)):
        adjacent = cosine_similarity(vectors[index - 1], vectors[index])
        left = mean_vector(vectors[max(0, index - window) : index])
        right = mean_vector(vectors[index : min(len(vectors), index + window)])
        contextual = cosine_similarity(left, right)
        scores.append(0.4 * adjacent + 0.6 * contextual)
    return scores


def is_local_semantic_low(
    scores: list[float | None], index: int, *, min_drop: float,
) -> bool:
    score = scores[index]
    if score is None:
        return False
    neighbors = [
        value
        for neighbor in (index - 1, index + 1)
        if 1 <= neighbor < len(scores)
        if (value := scores[neighbor]) is not None
    ]
    if len(neighbors) < 2 or score > min(neighbors):
        return False
    return (sum(neighbors) / len(neighbors)) - score >= min_drop


def expand_oversized_paragraphs(units: list[TextUnit], max_tokens: int) -> list[TextUnit]:
    expanded: list[TextUnit] = []
    maximum = max(max_tokens, 32)
    for unit in units:
        if estimate_text_tokens(unit.text) <= maximum:
            expanded.append(unit)
            continue
        sentence_units = []
        for sentence in split_sentences(unit.text):
            sentence_units.extend(TextUnit(
                fragment, unit.page, unit.section,
                paragraph_id=unit.paragraph_id, block_type=unit.block_type,
            ) for fragment in split_oversized_text(sentence, maximum))
        expanded.extend(
            TextUnit(
                " ".join(item.text for item in window), unit.page, unit.section,
                paragraph_id=unit.paragraph_id, block_type=unit.block_type,
            )
            for window in make_windows(sentence_units, maximum, 0, max_tokens=maximum)
        )
    return expanded


def split_oversized_text(text: str, max_tokens: int) -> list[str]:
    if estimate_text_tokens(text) <= max_tokens:
        return [text]
    words = text.split()
    if len(words) <= 1:
        character_limit = max(max_tokens, 1)
        return [text[index : index + character_limit] for index in range(0, len(text), character_limit)]
    fragments: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and estimate_text_tokens(candidate) > max_tokens:
            fragments.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        fragments.append(" ".join(current))
    return fragments


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


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = min(len(vector) for vector in vectors)
    if dimensions == 0:
        return []
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimensions)
    ]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(quantile, 1.0)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if not first or not second:
        return 0.0
    dot = sum(a * b for a, b in zip(first, second, strict=False))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot / (first_norm * second_norm)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip(":"))
