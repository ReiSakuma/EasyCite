"""Local BibTeX discovery and lightweight ranking for EasyCite."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .schemas import CitationSuggestion, EvidenceSnippet


SKIP_DIRS = {
    ".git",
    ".venv",
    "dist",
    "node_modules",
    "out",
    "__pycache__",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "methods",
    "method",
    "model",
    "models",
    "multi",
    "of",
    "on",
    "or",
    "our",
    "paper",
    "prior",
    "previous",
    "recent",
    "rely",
    "rely on",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "uses",
    "using",
    "we",
    "with",
}


@dataclass(frozen=True)
class BibEntry:
    key: str
    entry_type: str
    fields: dict[str, str] = field(default_factory=dict)
    bib_path: str | None = None

    @property
    def title(self) -> str:
        return self.fields.get("title", "").strip()

    @property
    def authors(self) -> list[str]:
        author_field = self.fields.get("author", "")
        return [part.strip() for part in re.split(r"\s+and\s+", author_field) if part.strip()]

    @property
    def year(self) -> str | None:
        return self.fields.get("year") or self.fields.get("date")

    @property
    def venue(self) -> str | None:
        return (
            self.fields.get("booktitle")
            or self.fields.get("journal")
            or self.fields.get("conference")
            or self.fields.get("publisher")
        )


class BibRepository:
    """Find and rank BibTeX entries relevant to a selected claim."""

    def __init__(self, workspace_path: str, file_path: str | None = None):
        self.workspace_path = Path(workspace_path).expanduser() if workspace_path else None
        self.file_path = Path(file_path).expanduser() if file_path else None

    def search(self, claim: str, limit: int = 5) -> list[CitationSuggestion]:
        query_terms = tokenize(claim)
        if not query_terms:
            return []

        ranked: list[tuple[float, BibEntry, str]] = []
        for entry in self.load_entries():
            score, reason = rank_entry(query_terms, entry)
            if score > 0:
                ranked.append((score, entry, reason))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            entry_to_suggestion(entry, score=score, reason=reason)
            for score, entry, reason in ranked[:limit]
        ]

    def load_entries(self) -> list[BibEntry]:
        entries: list[BibEntry] = []
        for bib_path in self.find_bib_files():
            try:
                content = bib_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for entry in parse_bibtex(content):
                entries.append(
                    BibEntry(
                        key=entry.key,
                        entry_type=entry.entry_type,
                        fields=entry.fields,
                        bib_path=os.fspath(bib_path),
                    )
                )
        return dedupe_entries(entries)

    def find_bib_files(self) -> list[Path]:
        referenced = self._bib_files_referenced_by_tex()
        if referenced:
            return referenced

        root = self.workspace_path
        if not root or not root.is_dir():
            root = self.file_path.parent if self.file_path and self.file_path.parent else None
        if not root:
            return []

        found: list[Path] = []
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
            for filename in filenames:
                if filename.lower().endswith(".bib"):
                    found.append(Path(current_root) / filename)
        return sorted(found)

    def _bib_files_referenced_by_tex(self) -> list[Path]:
        if not self.file_path or not self.file_path.exists():
            return []

        try:
            content = self.file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        base_dir = self.file_path.parent
        paths: list[Path] = []

        for match in re.finditer(r"\\addbibresource(?:\[[^\]]*\])?\{([^}]+)\}", content):
            paths.extend(resolve_bib_paths(match.group(1), base_dir))

        for match in re.finditer(r"\\bibliography\{([^}]+)\}", content):
            for part in match.group(1).split(","):
                paths.extend(resolve_bib_paths(part.strip(), base_dir))

        return [path for path in paths if path.exists()]


def resolve_bib_paths(raw_path: str, base_dir: Path) -> list[Path]:
    if not raw_path:
        return []
    path = Path(raw_path)
    if not path.suffix:
        path = path.with_suffix(".bib")
    if not path.is_absolute():
        path = base_dir / path
    return [path]


def parse_bibtex(content: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    index = 0
    while index < len(content):
        at_index = content.find("@", index)
        if at_index == -1:
            break
        open_index = content.find("{", at_index)
        if open_index == -1:
            break

        entry_type = content[at_index + 1 : open_index].strip().lower()
        close_index = find_matching_brace(content, open_index)
        if close_index == -1:
            break

        body = content[open_index + 1 : close_index]
        if entry_type not in {"comment", "preamble", "string"}:
            entry = parse_entry_body(entry_type, body)
            if entry:
                entries.append(entry)
        index = close_index + 1

    return entries


def find_matching_brace(content: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(content)):
        char = content[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def parse_entry_body(entry_type: str, body: str) -> BibEntry | None:
    comma_index = body.find(",")
    if comma_index == -1:
        return None
    key = body[:comma_index].strip()
    if not key:
        return None

    fields = parse_fields(body[comma_index + 1 :])
    return BibEntry(key=key, entry_type=entry_type, fields=fields)


def parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    index = 0
    while index < len(text):
        name_match = re.search(r"([A-Za-z][A-Za-z0-9_\-]*)\s*=", text[index:])
        if not name_match:
            break

        name = name_match.group(1).lower()
        value_start = index + name_match.end()
        value, value_end = parse_field_value(text, value_start)
        fields[name] = normalize_bib_value(value)
        index = value_end

    return fields


def parse_field_value(text: str, start: int) -> tuple[str, int]:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1

    if index >= len(text):
        return "", index

    delimiter = text[index]
    if delimiter == "{":
        end = find_matching_brace(text, index)
        if end == -1:
            return text[index + 1 :], len(text)
        return text[index + 1 : end], end + 1

    if delimiter == '"':
        end = index + 1
        while end < len(text):
            if text[end] == '"' and text[end - 1] != "\\":
                return text[index + 1 : end], end + 1
            end += 1
        return text[index + 1 :], len(text)

    end = index
    while end < len(text) and text[end] not in {","}:
        end += 1
    return text[index:end], end + 1


def normalize_bib_value(value: str) -> str:
    value = re.sub(r"[{}]", "", value)
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", cleaned.lower())
    terms = {word.strip("-") for word in words if word not in STOPWORDS}
    expanded: set[str] = set()
    for term in terms:
        expanded.add(term)
        if "-" in term:
            expanded.update(part for part in term.split("-") if len(part) > 2)
    return expanded


def rank_entry(query_terms: set[str], entry: BibEntry) -> tuple[float, str]:
    title_terms = tokenize(entry.title)
    abstract_terms = tokenize(entry.fields.get("abstract", ""))
    keyword_terms = tokenize(entry.fields.get("keywords", ""))
    venue_terms = tokenize(entry.venue or "")

    title_hits = query_terms & title_terms
    abstract_hits = query_terms & abstract_terms
    keyword_hits = query_terms & keyword_terms
    venue_hits = query_terms & venue_terms

    weighted = (
        len(title_hits) * 2.0
        + len(abstract_hits) * 1.1
        + len(keyword_hits) * 1.5
        + len(venue_hits) * 0.6
    )
    denominator = max(len(query_terms), 1)
    score = min(0.95, weighted / denominator)

    hits = sorted(title_hits | abstract_hits | keyword_hits | venue_hits)
    reason = (
        f"与选中文本共享关键词：{', '.join(hits[:8])}。"
        if hits
        else "BibTeX 元数据与选中文本存在弱相关。"
    )
    return score, reason


def entry_to_suggestion(entry: BibEntry, score: float, reason: str) -> CitationSuggestion:
    evidence_text = entry.fields.get("abstract") or reason
    evidence_source = "abstract" if entry.fields.get("abstract") else "metadata"
    return CitationSuggestion(
        paper_id=f"bib:{entry.key}",
        title=entry.title or entry.key,
        citation_key=entry.key,
        authors=entry.authors[:8],
        year=entry.year,
        venue=entry.venue,
        doi=entry.fields.get("doi"),
        url=entry.fields.get("url"),
        source="current_bib",
        verdict="metadata_match",
        confidence=max(0.25, min(score, 0.95)),
        evidence=EvidenceSnippet(
            section="BibTeX abstract" if evidence_source == "abstract" else "BibTeX metadata",
            source=evidence_source,
            text=evidence_text,
        ),
        reason=reason,
    )


def dedupe_entries(entries: Iterable[BibEntry]) -> list[BibEntry]:
    seen: set[str] = set()
    deduped: list[BibEntry] = []
    for entry in entries:
        if entry.key in seen:
            continue
        seen.add(entry.key)
        deduped.append(entry)
    return deduped
