"""Resolve missing BibTeX entries for accepted citation suggestions."""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .config import SourceConfig
from .tools.adapters import execute_better_bibtex_export
from .tools.errors import ToolExecutionError
from .zotero import ZoteroClient


def resolve_bibtex_entries(
    params: dict[str, Any],
    source_config: SourceConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return BibTeX entries for citation keys that are not already in the project."""
    source_config = source_config or SourceConfig.from_env()
    citation_keys = normalize_key_list(params.get("citation_keys"))
    suggestions = params.get("suggestions") if isinstance(params.get("suggestions"), list) else []
    suggestion_by_key = {
        str(suggestion.get("citation_key")).strip(): suggestion
        for suggestion in suggestions
        if isinstance(suggestion, dict) and suggestion.get("citation_key")
    }

    entries: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    exported_by_key: dict[str, str] = {}

    if citation_keys and source_config.zotero_enabled and source_config.zotero_endpoint:
        client = ZoteroClient(
            endpoint=source_config.zotero_endpoint,
            timeout_seconds=source_config.zotero_timeout_seconds,
            logger=logger,
        )
        try:
            exported = execute_better_bibtex_export(client, citation_keys)
        except ToolExecutionError as exc:
            exported = None
            if logger:
                logger(
                    "Better BibTeX tool failed; candidate metadata fallback retained: "
                    f"category={exc.category}, error={exc}"
                )
        exported_by_key = split_bibtex_entries(exported or "")
    elif citation_keys and logger:
        logger("Better BibTeX export skipped: Zotero is disabled or endpoint is empty.")

    for key in citation_keys:
        exported_bibtex = exported_by_key.get(key)
        if exported_bibtex and is_valid_bibtex_entry_for_key(exported_bibtex, key):
            entries.append(
                resolved_entry(
                    citation_key=key,
                    bibtex=exported_bibtex,
                    source="better-bibtex:zotero",
                )
            )
            continue
        if exported_bibtex and logger:
            logger(f"Better BibTeX export ignored for @{key}: exported entry key mismatch or malformed BibTeX.")

        suggestion = suggestion_by_key.get(key)
        fallback_bibtex = suggestion_to_fallback_bibtex(key, suggestion)
        if fallback_bibtex:
            entries.append(
                resolved_entry(
                    citation_key=key,
                    bibtex=fallback_bibtex,
                    source=f"candidate-metadata-fallback:{suggestion.get('source') or 'unknown'}"
                    if isinstance(suggestion, dict)
                    else "candidate-metadata-fallback:unknown",
                )
            )
            continue

        if source_config.zotero_enabled and source_config.zotero_endpoint:
            reason = "Better BibTeX 未返回该 citation key，且候选元数据不足，无法生成兜底 BibTeX。"
        else:
            reason = "Zotero 未启用或 endpoint 为空，且候选元数据不足，无法生成兜底 BibTeX。"
        missing.append(
            {
                "citation_key": key,
                "reason": reason,
            }
        )

    if logger:
        logger(
            "BibTeX resolver suggestion sources: "
            + ", ".join(
                f"@{key}:{suggestion_source_label(suggestion_by_key.get(key))}"
                for key in citation_keys
            )
        )
        logger(
            "Resolved BibTeX entries: "
            f"requested={len(citation_keys)}, entries={len(entries)}, missing={len(missing)}."
        )

    entries.sort(key=lambda entry: citation_keys.index(entry["citation_key"]))
    missing.sort(key=lambda item: citation_keys.index(item["citation_key"]))
    return {
        "entries": entries,
        "missing": missing,
    }


def resolved_entry(citation_key: str, bibtex: str, source: str) -> dict[str, Any]:
    """Return a resolved BibTeX entry with explicit provenance and trust metadata."""
    trust = bibtex_source_trust(source)
    warnings: list[str] = []
    if trust == "low":
        warnings.append("该 BibTeX 由候选元数据兜底生成，提交论文前必须人工核对。")
    if source.startswith("candidate-metadata-fallback:"):
        warnings.append("未能从 Better BibTeX/外部元数据源导出标准条目。")
    return {
        "citation_key": citation_key,
        "bibtex": bibtex,
        "source": source,
        "source_trust": trust,
        "source_rank": bibtex_source_rank(source),
        "requires_user_verification": trust != "high",
        "quality_warnings": warnings,
    }


def bibtex_source_rank(source: str) -> int:
    """Rank BibTeX sources according to the v14 provenance contract."""
    if source == "better-bibtex:zotero":
        return 1
    if source.startswith("zotero-metadata:"):
        return 2
    if source.startswith(("doi-metadata:", "arxiv-metadata:")):
        return 3
    if source.startswith("candidate-metadata-fallback:"):
        return 4
    return 99


def bibtex_source_trust(source: str) -> str:
    rank = bibtex_source_rank(source)
    if rank == 1:
        return "high"
    if rank in {2, 3}:
        return "medium"
    return "low"


def normalize_key_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    keys: list[str] = []
    for item in value:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def is_zotero_suggestion(suggestion: Any) -> bool:
    if not isinstance(suggestion, dict):
        return False
    source = clean_text(suggestion.get("source")).lower()
    paper_id = clean_text(suggestion.get("paper_id")).lower()
    return source == "zotero" or paper_id.startswith("zotero:")


def suggestion_source_label(suggestion: Any) -> str:
    if not isinstance(suggestion, dict):
        return "<missing-suggestion>"
    source = clean_text(suggestion.get("source")) or "<no-source>"
    paper_id = clean_text(suggestion.get("paper_id")) or "<no-paper-id>"
    title = clean_text(suggestion.get("title"))[:60] or "<no-title>"
    return f"source={source}, paper_id={paper_id}, title={title}"


def suggestion_to_fallback_bibtex(citation_key: str, suggestion: Any) -> str | None:
    """Create a conservative BibTeX entry from already-known candidate metadata."""
    if not isinstance(suggestion, dict):
        return None

    title = clean_text(suggestion.get("title"))
    if not title:
        return None

    authors = normalize_authors(suggestion.get("authors"))
    year = clean_text(suggestion.get("year"))
    venue = clean_text(suggestion.get("venue"))
    doi = clean_text(suggestion.get("doi"))
    url = clean_text(suggestion.get("url"))
    arxiv_id = extract_arxiv_id(" ".join([url, doi, title, venue]))
    source = clean_text(suggestion.get("source")) or "unknown"
    entry_type = fallback_entry_type(source=source, venue=venue, doi=doi, url=url)
    fields: list[tuple[str, str]] = [("title", title)]

    if authors:
        fields.append(("author", " and ".join(authors)))
    if year:
        fields.append(("year", year))
    if venue:
        fields.append((venue_field_for_entry_type(entry_type), venue))
    if doi:
        fields.append(("doi", doi))
    if arxiv_id:
        fields.append(("eprint", arxiv_id))
        fields.append(("archivePrefix", "arXiv"))
    if url:
        fields.append(("url", url))
    fields.append(("note", "Candidate-metadata fallback generated by EasyCite; verify before submission."))

    body = ",\n".join(
        f"  {name} = {{{escape_bibtex_value(value)}}}"
        for name, value in fields
        if value
    )
    return f"@{entry_type}{{{citation_key},\n{body}\n}}"


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for author in value:
        text = clean_text(author)
        if text:
            authors.append(text)
    return authors


def fallback_entry_type(source: str, venue: str, doi: str, url: str) -> str:
    lowered_venue = venue.lower()
    lowered_url = url.lower()
    if "conference" in lowered_venue or "proceedings" in lowered_venue:
        return "inproceedings"
    if "journal" in lowered_venue or doi:
        return "article"
    if "arxiv" in lowered_url or "arxiv" in lowered_venue or source == "online":
        return "misc"
    return "misc"


def venue_field_for_entry_type(entry_type: str) -> str:
    if entry_type == "inproceedings":
        return "booktitle"
    if entry_type == "article":
        return "journal"
    return "howpublished"


def escape_bibtex_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def split_bibtex_entries(content: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    index = 0
    while index < len(content):
        at_index = content.find("@", index)
        if at_index == -1:
            break
        open_index = content.find("{", at_index)
        if open_index == -1:
            break
        close_index = find_matching_brace(content, open_index)
        if close_index == -1:
            break
        key_match = re.match(r"\s*([^,\s]+)\s*,", content[open_index + 1 : close_index])
        if key_match:
            key = key_match.group(1)
            entries[key] = content[at_index : close_index + 1].strip()
        index = close_index + 1
    return entries


def is_valid_bibtex_entry_for_key(entry: str, citation_key: str) -> bool:
    entries = split_bibtex_entries(entry)
    return citation_key in entries and bool(entries[citation_key].strip())


def extract_arxiv_id(text: str) -> str:
    match = re.search(r"(?:arxiv[:/ ]+|abs/)(\d{4}\.\d{4,5})(?:v\d+)?", text, flags=re.I)
    return match.group(1) if match else ""


def find_matching_brace(content: str, open_index: int) -> int:
    depth = 0
    escaped = False
    for index in range(open_index, len(content)):
        char = content[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1
