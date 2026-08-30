"""Configurable read-only Zotero candidate adapter.

V2 keeps Zotero optional. It supports both the official Zotero Local API and
the simplified mock/bridge endpoint used for manual testing.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .schemas import CitationSuggestion, EvidenceSnippet
from .tools.errors import ToolExecutionError


ALLOWED_ITEM_TYPES = {
    "book",
    "bookSection",
    "conferencePaper",
    "journalArticle",
    "magazineArticle",
    "manuscript",
    "preprint",
    "report",
    "thesis",
}


class ZoteroClient:
    """Search a configurable Zotero-compatible HTTP endpoint."""

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = 4.0,
        collection_keys: list[str] | None = None,
        logger: Callable[[str], None] | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.collection_keys = [key.strip() for key in (collection_keys or []) if key.strip()]
        self.logger = logger
        self.last_error: BaseException | None = None

    def clear_last_error(self) -> None:
        self.last_error = None

    def search(
        self,
        claim: str,
        limit: int = 5,
        query_phrases: list[str] | None = None,
        metadata_limit: int | None = None,
    ) -> list[CitationSuggestion]:
        if not self.endpoint:
            self._log("Zotero endpoint is empty; skipping Zotero search.")
            return []

        if self._looks_like_zotero_api():
            return self._search_zotero_api(
                claim=claim,
                limit=limit,
                query_phrases=query_phrases,
                metadata_limit=metadata_limit,
            )

        payload = self._request_json("/search", claim, limit)
        if payload is None:
            payload = self._request_json("/items", claim, limit)
        if payload is None:
            self._log("Zotero bridge search returned no readable JSON payload.")
            return []

        items = payload if isinstance(payload, list) else payload.get("items", [])
        self._log(f"Zotero bridge returned {len(items)} raw item(s).")
        suggestions: list[CitationSuggestion] = []
        for item in sorted(items, key=zotero_item_sort_key):
            suggestion = zotero_item_to_suggestion(item)
            if suggestion:
                suggestions.append(suggestion)
            if len(suggestions) >= limit:
                break
        return suggestions

    def _looks_like_zotero_api(self) -> bool:
        parsed = urllib.parse.urlparse(self.endpoint)
        return parsed.path.rstrip("/") == "/api" or parsed.netloc.endswith(":23119")

    def _search_zotero_api(
        self,
        claim: str,
        limit: int,
        query_phrases: list[str] | None = None,
        metadata_limit: int | None = None,
    ) -> list[CitationSuggestion]:
        api_root = self.endpoint.rstrip("/")
        if not api_root.endswith("/api"):
            api_root = f"{api_root}/api"

        queries = dedupe_queries(query_phrases or [claim])[:32]
        max_raw = max(metadata_limit or limit, limit)
        per_query_limit = max(5, min(12, max_raw))
        raw_cap = max(max_raw, min(max_raw * 3, 180))
        merged_items: dict[str, Any] = {}
        query_hits: list[tuple[str, int]] = []
        scope_paths = zotero_search_paths(self.collection_keys)
        if self.collection_keys:
            self._log(
                "Zotero Local API scope limited to collection key(s): "
                + ", ".join(self.collection_keys)
            )

        for query_text in queries:
            query = urllib.parse.urlencode(
                {
                    "q": query_text,
                    "qmode": "everything",
                    "limit": str(per_query_limit),
                    "format": "json",
                }
            )
            total_query_items = 0
            for path in scope_paths:
                url = f"{api_root}{path}?{query}"
                payload = self._request_url_json(url)
                if payload is None:
                    self._log(f"Zotero Local API query returned no JSON: {query_text}")
                    continue

                items = payload if isinstance(payload, list) else payload.get("items", [])
                total_query_items += len(items)
                scope_label = "whole library" if path == "/users/0/items/top" else path
                self._log(
                    "Zotero Local API query "
                    f"{query_text!r} in {scope_label} returned {len(items)} raw item(s)."
                )
                for item in items:
                    item_key = extract_item_key(item) or zotero_title(item)
                    if not item_key or item_key in merged_items:
                        continue
                    merged_items[item_key] = item
                    if len(merged_items) >= raw_cap:
                        break
                if len(merged_items) >= raw_cap:
                    break
            query_hits.append((query_text, total_query_items))
            if len(merged_items) >= raw_cap:
                break

        items = [
            item
            for item in merged_items.values()
            if is_citable_zotero_item(item)
        ]
        items = items[:raw_cap]
        self._log(
            "Zotero Local API returned "
            f"{len(merged_items)} raw unique item(s), {len(items)} citable item(s) "
            f"across {len(queries)} query phrase(s); hits={query_hits[:20]}."
        )
        bbt_citation_keys = self._fetch_better_bibtex_citation_keys(items)
        suggestions: list[CitationSuggestion] = []
        for item in items:
            item_key = extract_item_key(item)
            pdf_path = self._resolve_pdf_path(item)
            suggestion = zotero_item_to_suggestion(
                item,
                bbt_citation_key=bbt_citation_keys.get(item_key or ""),
                pdf_path=pdf_path,
            )
            if suggestion:
                suggestions.append(suggestion)
            if len(suggestions) >= raw_cap:
                break
        return suggestions

    def _fetch_better_bibtex_citation_keys(self, items: list[Any]) -> dict[str, str]:
        item_keys = [key for item in items if (key := extract_item_key(item))]
        if not item_keys:
            self._log("Better BibTeX citation key lookup skipped: no Zotero item keys.")
            return {}

        result = self._request_better_bibtex_json_rpc("item.citationkey", [item_keys])
        if not isinstance(result, dict):
            self._log("Better BibTeX citation key lookup returned no usable mapping.")
            return {}

        usable = {
            str(item_key): str(citation_key)
            for item_key, citation_key in result.items()
            if citation_key
        }
        self._log(
            f"Better BibTeX citation key lookup returned {len(usable)} key(s) "
            f"for {len(item_keys)} Zotero item(s)."
        )
        return usable

    def export_bibtex(
        self,
        citation_keys: list[str],
        translator: str = "Better BibTeX",
    ) -> str | None:
        keys = [key for key in citation_keys if key]
        if not keys:
            self._log("Better BibTeX export skipped: no citation keys.")
            return None

        result = self._request_better_bibtex_json_rpc(
            "item.export",
            [keys, translator],
        )
        if isinstance(result, str) and result.strip():
            self._log(
                f"Better BibTeX export returned {len(result)} character(s) "
                f"for {len(keys)} citation key(s)."
            )
            return result

        self._log("Better BibTeX export returned no usable BibTeX string.")
        return None

    def _request_better_bibtex_json_rpc(
        self, method: str, params: list[Any]
    ) -> Any | None:
        rpc_url = better_bibtex_rpc_url(self.endpoint)
        self._log(f"Requesting Better BibTeX JSON-RPC method: {method}")
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": "easycite-v20",
        }
        request = urllib.request.Request(
            rpc_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Citation-Agent-v2",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                self._log(f"Better BibTeX HTTP response: {response.status}")
                data = json.loads(response.read().decode("utf-8"))
                self.last_error = None
        except urllib.error.HTTPError as exc:
            self.last_error = exc
            self._log(f"Better BibTeX HTTP error: {exc.code} {exc.reason}")
            return None
        except urllib.error.URLError as exc:
            self.last_error = exc
            self._log(f"Better BibTeX connection error: {exc.reason}")
            return None
        except OSError as exc:
            self.last_error = exc
            self._log(f"Better BibTeX OS/network error: {exc}")
            return None
        except json.JSONDecodeError as exc:
            self.last_error = ToolExecutionError(
                "invalid_output",
                f"Better BibTeX returned invalid JSON: {exc}",
            )
            self._log(f"Better BibTeX JSON decode error: {exc}")
            return None

        if isinstance(data, dict) and data.get("error"):
            self.last_error = ToolExecutionError(
                "invalid_output",
                f"Better BibTeX JSON-RPC error: {data['error']}",
            )
            self._log(f"Better BibTeX JSON-RPC error: {data['error']}")
            return None
        return data.get("result") if isinstance(data, dict) else None

    def _request_json(self, path: str, claim: str, limit: int) -> Any | None:
        query = urllib.parse.urlencode({"q": claim, "limit": str(limit)})
        url = f"{self.endpoint}{path}?{query}"
        return self._request_url_json(url)

    def _request_url_json(self, url: str) -> Any | None:
        self._log(f"Requesting Zotero URL: {redact_query(url)}")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Citation-Agent-v2",
                "Zotero-API-Version": "3",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                self._log(f"Zotero HTTP response: {response.status}")
                value = json.loads(response.read().decode("utf-8"))
                self.last_error = None
                return value
        except urllib.error.HTTPError as exc:
            self.last_error = exc
            self._log(f"Zotero HTTP error: {exc.code} {exc.reason}")
            return None
        except urllib.error.URLError as exc:
            self.last_error = exc
            self._log(f"Zotero connection error: {exc.reason}")
            return None
        except OSError as exc:
            self.last_error = exc
            self._log(f"Zotero OS/network error: {exc}")
            return None
        except json.JSONDecodeError as exc:
            self.last_error = ToolExecutionError(
                "invalid_output",
                f"Zotero returned invalid JSON: {exc}",
            )
            self._log(f"Zotero JSON decode error: {exc}")
            return None

    def _resolve_pdf_path(self, item: Any) -> str | None:
        direct = zotero_enclosure_path(item)
        if direct:
            return direct

        attachment_url = zotero_attachment_url(item)
        if not attachment_url:
            return None

        payload = self._request_url_json(attachment_url)
        if payload is None:
            return None
        if isinstance(payload, list):
            for child in payload:
                child_path = zotero_enclosure_path(child)
                if child_path:
                    return child_path
        return zotero_enclosure_path(payload)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)


def redact_query(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    redacted = {
        key: ["<claim>" if key == "q" else value for value in values]
        for key, values in query.items()
    }
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(redacted, doseq=True))
    )


def dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        normalized = re.sub(r"\s+", " ", str(query or "")).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def better_bibtex_rpc_url(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    return urllib.parse.urlunparse(
        parsed._replace(path="/better-bibtex/json-rpc", params="", query="", fragment="")
    )


def zotero_search_paths(collection_keys: list[str]) -> list[str]:
    if not collection_keys:
        return ["/users/0/items/top"]
    return [
        f"/users/0/collections/{urllib.parse.quote(key, safe='')}/items/top"
        for key in collection_keys
        if key
    ]


def extract_item_key(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None

    data = item.get("data") if isinstance(item.get("data"), dict) else item
    key = data.get("key") or item.get("key")
    if not key:
        return None

    library_id = data.get("libraryID") or item.get("libraryID")
    if library_id:
        return f"{library_id}:{key}"
    return str(key)


def zotero_item_to_suggestion(
    item: Any,
    bbt_citation_key: str | None = None,
    pdf_path: str | None = None,
) -> CitationSuggestion | None:
    if not isinstance(item, dict):
        return None

    data = item.get("data") if isinstance(item.get("data"), dict) else item
    title = str(data.get("title") or item.get("title") or "").strip()
    if not title:
        return None

    citation_key = bbt_citation_key or extract_citation_key(data, item)
    creators = data.get("creators") or data.get("authors") or item.get("authors") or []
    authors = normalize_authors(creators)
    abstract = data.get("abstractNote") or data.get("abstract") or item.get("abstract")
    evidence_section = "Zotero abstract" if abstract else "Zotero metadata"

    return CitationSuggestion(
        paper_id=f"zotero:{data.get('key') or item.get('key') or title}",
        title=title,
        citation_key=str(citation_key) if citation_key else None,
        authors=authors[:8],
        year=extract_year(data.get("date") or data.get("year") or item.get("year")),
        venue=data.get("publicationTitle")
        or data.get("conferenceName")
        or data.get("proceedingsTitle")
        or item.get("venue"),
        doi=data.get("DOI") or data.get("doi") or item.get("doi"),
        url=data.get("url") or item.get("url"),
        pdf_path=pdf_path
        or data.get("pdfPath")
        or item.get("pdfPath")
        or item.get("pdf_path"),
        source="zotero",
        verdict="metadata_match",
        confidence=0.68 if citation_key else 0.56,
        evidence=EvidenceSnippet(
            section=evidence_section,
            source="abstract" if abstract else "metadata",
            text=abstract or title,
        ),
        reason="来自配置的 Zotero 只读搜索端点；等待 evidence passage 验证。",
    )


def zotero_title(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    data = item.get("data") if isinstance(item.get("data"), dict) else item
    return str(data.get("title") or item.get("title") or "").strip()


def zotero_item_sort_key(item: Any) -> tuple[str, str, str]:
    if not isinstance(item, dict):
        return ("", "", "")
    data = item.get("data") if isinstance(item.get("data"), dict) else item
    title = re.sub(r"[^a-z0-9]+", " ", zotero_title(item).lower()).strip()
    year = extract_year(data.get("date") or data.get("year") or item.get("year")) or ""
    key = str(data.get("key") or item.get("key") or "")
    return (title, year, key)


def is_citable_zotero_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    data = item.get("data") if isinstance(item.get("data"), dict) else item
    item_type = data.get("itemType") or item.get("itemType")
    if item_type and item_type not in ALLOWED_ITEM_TYPES:
        return False

    title = zotero_title(item)
    if len(title) < 5:
        return False
    lowered = re.sub(r"\W+", "", title.lower())
    if lowered in {"pdf", "fulltext", "supplementarymaterial"}:
        return False
    return True


def zotero_attachment_url(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    links = item.get("links") if isinstance(item.get("links"), dict) else {}
    attachment = links.get("attachment") if isinstance(links.get("attachment"), dict) else {}
    href = attachment.get("href")
    return str(href) if href else None


def zotero_enclosure_path(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None

    links = item.get("links") if isinstance(item.get("links"), dict) else {}
    enclosure = links.get("enclosure") if isinstance(links.get("enclosure"), dict) else {}
    href = enclosure.get("href")
    if not href:
        return None

    parsed = urllib.parse.urlparse(str(href))
    if parsed.scheme != "file":
        return None
    return normalize_file_uri_path(urllib.request.url2pathname(parsed.path))


def normalize_file_uri_path(path: str) -> str:
    if re.match(r"^/[A-Za-z]:/", path):
        drive = path[1].lower()
        rest = path[3:]
        return os.path.join("/mnt", drive, rest.lstrip("/"))
    if re.match(r"^[A-Za-z]:/", path):
        drive = path[0].lower()
        rest = path[2:]
        return os.path.join("/mnt", drive, rest.lstrip("/\\"))
    return path


def normalize_authors(creators: Any) -> list[str]:
    if not isinstance(creators, list):
        return []

    authors: list[str] = []
    for creator in creators:
        if isinstance(creator, str):
            authors.append(creator)
            continue
        if not isinstance(creator, dict):
            continue
        name = creator.get("name")
        if not name:
            first = creator.get("firstName") or creator.get("given") or ""
            last = creator.get("lastName") or creator.get("family") or ""
            name = f"{first} {last}".strip()
        if name:
            authors.append(str(name))
    return authors


def extract_citation_key(data: dict[str, Any], item: dict[str, Any]) -> str | None:
    direct_key = (
        data.get("citationKey")
        or data.get("citekey")
        or data.get("betterBibTeXKey")
        or item.get("citationKey")
        or item.get("citekey")
    )
    if direct_key:
        return str(direct_key)

    extra = data.get("extra") or item.get("extra")
    if isinstance(extra, str):
        match = re.search(r"^\s*Citation Key:\s*(\S+)\s*$", extra, flags=re.MULTILINE)
        if match:
            return match.group(1)

    return None


def extract_year(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    for index in range(0, max(len(text) - 3, 0)):
        candidate = text[index : index + 4]
        if candidate.isdigit():
            return candidate
    return text if text.isdigit() else None
