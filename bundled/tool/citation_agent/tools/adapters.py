"""Adapters that expose EasyCite integrations through ToolExecutor."""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from .base import (
    IdempotencyPolicy,
    PermissionPolicy,
    RetryPolicy,
    TimeoutPolicy,
    ToolDefinition,
    ToolHealth,
)
from .errors import ToolExecutionError
from .executor import get_default_executor
from .registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="current_bib.search",
            version="1.0.0",
            description="Return candidates already retrieved from the current project bibliography.",
            input_schema={"fallback_candidates": list},
            required_fields=("fallback_candidates",),
            output_validator=lambda value: isinstance(value, list),
            capabilities=("citation_retrieval", "offline_fallback"),
            handler=lambda inputs: list(inputs["fallback_candidates"]),
        )
    )
    registry.register(
        ToolDefinition(
            name="zotero.search",
            version="1.0.0",
            description="Read-only search against Zotero Local API.",
            input_schema={"client": object, "claim": str, "limit": int, "fallback_candidates": list},
            required_fields=("client", "claim", "limit", "fallback_candidates"),
            output_validator=lambda value: isinstance(value, list),
            timeout_policy=TimeoutPolicy(12.0),
            retry_policy=RetryPolicy(max_retries=1, initial_backoff_seconds=0.15),
            fallback_chain=("current_bib.search",),
            capabilities=("citation_retrieval", "zotero", "read_only"),
            handler=_zotero_search_handler,
            health_check=_zotero_health,
        )
    )
    registry.register(
        ToolDefinition(
            name="better_bibtex.export",
            version="1.0.0",
            description="Export trusted BibTeX through Better BibTeX JSON-RPC.",
            input_schema={"client": object, "citation_keys": list},
            required_fields=("client", "citation_keys"),
            output_validator=lambda value: value is None or isinstance(value, str),
            timeout_policy=TimeoutPolicy(10.0),
            retry_policy=RetryPolicy(max_retries=1, initial_backoff_seconds=0.15),
            capabilities=("bibtex_export", "zotero", "read_only"),
            handler=_better_bibtex_export_handler,
            health_check=_better_bibtex_health,
        )
    )
    registry.register(
        ToolDefinition(
            name="llm.completion",
            version="1.0.0",
            description="Structured model completion through LiteLLM.",
            input_schema={"kwargs": dict},
            required_fields=("kwargs",),
            output_validator=_valid_llm_response,
            timeout_policy=TimeoutPolicy(65.0),
            retry_policy=RetryPolicy(max_retries=1, initial_backoff_seconds=0.2),
            capabilities=("llm", "structured_generation", "read_only"),
            handler=_llm_completion_handler,
            health_check=_llm_health,
        )
    )
    registry.register(
        ToolDefinition(
            name="pdf.read",
            version="1.0.0",
            description="Extract evidence passages from a local PDF attachment.",
            input_schema={"path": str},
            required_fields=("path",),
            output_validator=lambda value: isinstance(value, list),
            timeout_policy=TimeoutPolicy(30.0),
            retry_policy=RetryPolicy(max_retries=0),
            capabilities=("pdf", "evidence_extraction", "read_only"),
            health_check=_pdf_health,
        )
    )
    registry.register(
        ToolDefinition(
            name="embedding.rank",
            version="1.0.0",
            description="Compute local semantic vectors for retrieval and evidence ranking.",
            input_schema={"texts": (list, tuple)},
            required_fields=("texts",),
            output_validator=lambda value: hasattr(value, "vectors") and isinstance(value.vectors, list),
            timeout_policy=TimeoutPolicy(90.0),
            retry_policy=RetryPolicy(max_retries=0),
            capabilities=("embedding", "semantic_ranking", "read_only"),
            health_check=_embedding_health,
        )
    )
    registry.register(
        ToolDefinition(
            name="bibtex.write",
            version="1.0.0",
            description="Idempotently append validated BibTeX entries and verify the write.",
            input_schema={"path": str, "entries": list, "idempotency_key": str},
            required_fields=("path", "entries", "idempotency_key"),
            output_validator=_valid_bibtex_write_result,
            read_only=False,
            timeout_policy=TimeoutPolicy(10.0),
            retry_policy=RetryPolicy(max_retries=0),
            idempotency_policy=IdempotencyPolicy(enabled=True, require_key=True),
            permission_policy=PermissionPolicy(
                required_permission="workspace_bibtex_write",
                allow_by_default=False,
            ),
            capabilities=("bibtex_write", "workspace_mutation"),
            handler=_bibtex_write_handler,
            health_check=_bibtex_writer_health,
        )
    )
    return registry


def execute_litellm_completion(kwargs: dict[str, Any]) -> Any:
    timeout = float(kwargs.get("timeout") or 65.0) + 2.0
    return get_default_executor().execute(
        "llm.completion",
        {"kwargs": kwargs},
        timeout_seconds=timeout,
    ).value


def execute_zotero_search(
    client: Any,
    *,
    claim: str,
    limit: int,
    query_phrases: list[str] | None = None,
    metadata_limit: int | None = None,
    fallback_candidates: list[Any] | None = None,
) -> list[Any]:
    timeout = max(float(getattr(client, "timeout_seconds", 4.0)) * 2.5, 5.0)
    return get_default_executor().execute(
        "zotero.search",
        {
            "client": client,
            "claim": claim,
            "limit": int(limit),
            "query_phrases": list(query_phrases or []),
            "metadata_limit": metadata_limit,
            "fallback_candidates": list(fallback_candidates or []),
        },
        timeout_seconds=timeout,
    ).value


def execute_better_bibtex_export(client: Any, citation_keys: list[str]) -> str | None:
    timeout = max(float(getattr(client, "timeout_seconds", 4.0)) * 2.5, 5.0)
    return get_default_executor().execute(
        "better_bibtex.export",
        {"client": client, "citation_keys": citation_keys},
        timeout_seconds=timeout,
    ).value


def execute_pdf_reader(path: str, operation: Callable[[], list[Any]]) -> list[Any]:
    return get_default_executor().execute(
        "pdf.read",
        {"path": path},
        operation=lambda _inputs: operation(),
    ).value


def execute_embedding(texts: Sequence[str], operation: Callable[[], Any]) -> Any:
    return get_default_executor().execute(
        "embedding.rank",
        {"texts": list(texts)},
        operation=lambda _inputs: operation(),
    ).value


def execute_bibtex_write(
    path: str,
    entries: list[dict[str, str]],
    idempotency_key: str,
) -> dict[str, Any]:
    return get_default_executor().execute(
        "bibtex.write",
        {"path": path, "entries": entries, "idempotency_key": idempotency_key},
        permissions={"workspace_bibtex_write"},
        idempotency_key=idempotency_key,
    ).value


def _llm_completion_handler(inputs: dict[str, Any]) -> Any:
    from litellm import completion

    return completion(**inputs["kwargs"])


def _zotero_search_handler(inputs: dict[str, Any]) -> list[Any]:
    client = inputs["client"]
    if hasattr(client, "clear_last_error"):
        client.clear_last_error()
    result = client.search(
        inputs["claim"],
        limit=inputs["limit"],
        query_phrases=inputs.get("query_phrases"),
        metadata_limit=inputs.get("metadata_limit"),
    )
    last_error = getattr(client, "last_error", None)
    if last_error and not result:
        raise last_error
    return result


def _better_bibtex_export_handler(inputs: dict[str, Any]) -> str | None:
    client = inputs["client"]
    if hasattr(client, "clear_last_error"):
        client.clear_last_error()
    result = client.export_bibtex(inputs["citation_keys"])
    last_error = getattr(client, "last_error", None)
    if last_error and not result:
        raise last_error
    return result


def _valid_llm_response(value: Any) -> bool:
    choices = getattr(value, "choices", None)
    if not isinstance(choices, list) or not choices:
        return False
    message = getattr(choices[0], "message", None)
    return message is not None and getattr(message, "content", None) is not None


def _bibtex_write_handler(inputs: dict[str, Any]) -> dict[str, Any]:
    path = pathlib.Path(inputs["path"]).expanduser().resolve()
    entries = inputs["entries"]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_keys = set(re.findall(r"@\w+\s*[{(]\s*([^,\s]+)\s*,", existing))
    appended: list[str] = []
    skipped: list[str] = []
    blocks: list[str] = []
    for entry in entries:
        key = str(entry.get("citation_key") or "").strip()
        bibtex = str(entry.get("bibtex") or "").strip()
        parsed = re.match(r"@\w+\s*[{(]\s*([^,\s]+)\s*,", bibtex)
        if not key or not parsed or parsed.group(1) != key:
            raise ToolExecutionError("invalid_input", f"Malformed BibTeX entry for @{key or '<empty>'}.")
        if key in existing_keys:
            skipped.append(key)
            continue
        existing_keys.add(key)
        appended.append(key)
        blocks.append(bibtex)
    if blocks:
        separator = "\n\n" if existing.strip() else ""
        updated = existing.rstrip() + separator + "\n\n".join(blocks) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(updated)
            temp_path = pathlib.Path(handle.name)
        os.replace(temp_path, path)
    written = path.read_text(encoding="utf-8") if path.exists() else ""
    missing_after_write = [
        key
        for key in appended
        if not re.search(rf"@\w+\s*[{{(]\s*{re.escape(key)}\s*,", written)
    ]
    if missing_after_write:
        raise ToolExecutionError(
            "invalid_output",
            f"BibTeX write verification failed for: {', '.join(missing_after_write)}.",
        )
    return {"path": str(path), "appended": appended, "skipped_existing": skipped, "verified": True}


def _valid_bibtex_write_result(value: Any) -> bool:
    return isinstance(value, dict) and value.get("verified") is True and isinstance(value.get("appended"), list)


def _llm_health(context: dict[str, Any]) -> ToolHealth:
    config = context.get("model_config")
    if config is None or not config.use_llm:
        return ToolHealth("disabled", "模型 API 未启用。")
    missing = []
    if not config.api_key:
        missing.append("api key")
    if importlib.util.find_spec("litellm") is None:
        missing.append("litellm")
    if missing:
        return ToolHealth("error", f"模型工具缺少：{', '.join(missing)}。")
    return ToolHealth("ok", "LLM completion adapter 已注册且依赖可用。", (f"model={config.model}",))


def _zotero_health(context: dict[str, Any]) -> ToolHealth:
    config = context.get("source_config")
    if config is None or not config.zotero_enabled:
        return ToolHealth("disabled", "Zotero 候选源未启用。")
    if not config.zotero_endpoint:
        return ToolHealth("error", "Zotero endpoint 为空。")
    base = config.zotero_endpoint.rstrip("/")
    if not base.endswith("/api"):
        base += "/api"
    request = urllib.request.Request(
        f"{base}/users/0/items/top?limit=1&format=json",
        headers={"Zotero-API-Version": "3"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config.zotero_timeout_seconds) as response:
            if 200 <= response.status < 300:
                return ToolHealth("ok", "Zotero Local API 可访问。", (f"endpoint={base}",))
    except Exception as exc:  # pylint: disable=broad-except
        return ToolHealth(
            "error",
            "Zotero Local API 不可用；推荐将降级到当前 BibTeX。",
            (f"{type(exc).__name__}: {exc}",),
            ("启动 Zotero 或检查 Local API endpoint。",),
        )
    return ToolHealth("error", "Zotero Local API 返回异常状态。")


def _better_bibtex_health(context: dict[str, Any]) -> ToolHealth:
    config = context.get("source_config")
    if config is None or not config.zotero_enabled:
        return ToolHealth("disabled", "Zotero 未启用，因此不检查 Better BibTeX。")
    if not config.zotero_endpoint:
        return ToolHealth("error", "Zotero endpoint 为空。")
    from ..zotero import better_bibtex_rpc_url

    payload = json.dumps({"jsonrpc": "2.0", "method": "item.citationkey", "params": [[]], "id": "health"}).encode()
    request = urllib.request.Request(
        better_bibtex_rpc_url(config.zotero_endpoint),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.zotero_timeout_seconds) as response:
            if 200 <= response.status < 500:
                return ToolHealth("ok", "Better BibTeX JSON-RPC 可访问。")
    except Exception as exc:  # pylint: disable=broad-except
        return ToolHealth("warning", "Better BibTeX 不可用，将使用元数据兜底。", (str(exc),))
    return ToolHealth("warning", "Better BibTeX 返回异常状态。")


def _pdf_health(_context: dict[str, Any]) -> ToolHealth:
    if importlib.util.find_spec("fitz") is None:
        return ToolHealth("warning", "PyMuPDF 未安装，PDF 阅读将降级到摘要/元数据。")
    return ToolHealth("ok", "PDF Reader adapter 可用。", ("module=fitz",))


def _embedding_health(context: dict[str, Any]) -> ToolHealth:
    config = context.get("embedding_config")
    if config is None or not config.enabled:
        return ToolHealth("disabled", "本地 embedding 未启用。")
    from ..embedding.model_manager import EmbeddingModelManager

    status = EmbeddingModelManager(config).status()
    if status.available:
        return ToolHealth("ok", "Embedding adapter 可用。", (status.reason,))
    return ToolHealth("warning", "Embedding 不可用，将降级到词法排序。", (status.reason,))


def _bibtex_writer_health(context: dict[str, Any]) -> ToolHealth:
    workspace = pathlib.Path(str(context.get("workspace_path") or os.getcwd()))
    if not workspace.exists():
        return ToolHealth("error", "工作区不存在，BibTeX Writer 不可用。", (str(workspace),))
    bib_files = sorted(workspace.rglob("*.bib"))[:8]
    if not bib_files:
        return ToolHealth(
            "warning",
            "工作区尚无 .bib 文件；首次写入会在用户确认后创建 references.bib。",
            (str(workspace),),
        )
    return ToolHealth(
        "ok",
        "BibTeX Writer 已注册，写入时要求显式 workspace 权限。",
        tuple(str(path) for path in bib_files),
    )
