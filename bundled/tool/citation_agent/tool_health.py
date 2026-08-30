"""Registry-driven backend tool health checks for EasyCite."""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import EmbeddingConfig, ModelConfig, SourceConfig
from .tools.adapters import build_default_registry
from .tools.base import ToolHealth


ToolStatus = Literal["ok", "warning", "error", "disabled"]


@dataclass
class HealthCheckItem:
    name: str
    label: str
    status: ToolStatus
    summary: str
    details: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    latency_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
            "actions": self.actions,
            "latency_ms": self.latency_ms,
        }


def run_tool_health_check(
    params: dict[str, Any],
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Read the registered inventory and execute each declared health check."""

    workspace_path = str(params.get("workspace_path") or os.getcwd())
    file_path = str(params.get("file_path") or "")
    model_config = ModelConfig.from_env()
    source_config = SourceConfig.from_env()
    embedding_config = EmbeddingConfig.from_env()
    registry = build_default_registry()
    registry_health = registry.health(
        {
            "workspace_path": workspace_path,
            "file_path": file_path,
            "model_config": model_config,
            "source_config": source_config,
            "embedding_config": embedding_config,
        }
    )
    items = [
        check_python_runtime(),
        check_python_dependencies(),
        *[
            registry_health_item(name, health)
            for name, health in sorted(registry_health.items())
        ],
        check_memory(source_config, workspace_path),
    ]
    order = {"error": 0, "warning": 1, "disabled": 2, "ok": 3}
    items.sort(key=lambda item: (order[item.status], item.name))
    summary = summarize(items)
    if logger:
        logger(
            "Registry tool health completed: "
            f"overall={summary['overall_status']}, tools={len(registry.list())}, "
            f"ok={summary['ok']}, warning={summary['warning']}, "
            f"error={summary['error']}, disabled={summary['disabled']}."
        )
    return {
        "schema_version": "v17.tool_health.1",
        "workspace_path": workspace_path,
        "file_path": file_path,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "registry": [definition.describe() for definition in registry.list()],
        "items": [item.as_dict() for item in items],
    }


def registry_health_item(name: str, health: ToolHealth) -> HealthCheckItem:
    item_names = {
        "llm.completion": "llm",
        "zotero.search": "zotero",
        "better_bibtex.export": "better_bibtex",
        "pdf.read": "pdf_reader",
        "embedding.rank": "embedding",
        "bibtex.write": "bibtex_workspace",
    }
    labels = {
        "llm.completion": "Model API",
        "zotero.search": "Zotero Local API",
        "better_bibtex.export": "Better BibTeX",
        "pdf.read": "PDF Evidence Reader",
        "embedding.rank": "Embedding Model",
        "bibtex.write": "BibTeX Writer",
    }
    return HealthCheckItem(
        name=item_names.get(name, name),
        label=labels.get(name, name),
        status=health.status,
        summary=health.summary,
        details=list(health.details),
        actions=list(health.actions),
    )


def summarize(items: list[HealthCheckItem]) -> dict[str, Any]:
    counts = {
        status: sum(1 for item in items if item.status == status)
        for status in ("ok", "warning", "error", "disabled")
    }
    if counts["error"]:
        overall = "error"
        message = "有关键工具不可用，推荐链路可能失败或降级。"
    elif counts["warning"]:
        overall = "warning"
        message = "核心链路可运行，但部分增强能力会降级。"
    else:
        overall = "ok"
        message = "工具链状态良好。"
    return {**counts, "overall_status": overall, "message": message}


def check_python_runtime() -> HealthCheckItem:
    return HealthCheckItem(
        name="python",
        label="Python Runtime",
        status="ok",
        summary=f"Python {sys.version.split()[0]}",
        details=[f"executable={sys.executable}", f"platform={sys.platform}"],
    )


def check_python_dependencies() -> HealthCheckItem:
    modules = ["pydantic", "pydantic_core", "pygls", "lsprotocol", "langgraph"]
    missing = [module for module in modules if not module_available(module)]
    if missing:
        return HealthCheckItem(
            name="python_dependencies",
            label="Required Python Packages",
            status="error",
            summary=f"缺少必需包：{', '.join(missing)}",
            details=[f"sys.path={path}" for path in sys.path[:8]],
            actions=["运行 EasyCite: 设置向导，或允许安装到扩展私有目录。"],
        )
    return HealthCheckItem(
        name="python_dependencies",
        label="Required Python Packages",
        status="ok",
        summary="必需 Python 包均可导入。",
        details=[f"{module}=available" for module in modules],
    )


def check_memory(config: SourceConfig, workspace_path: str) -> HealthCheckItem:
    if not config.memory_enabled:
        return HealthCheckItem(
            name="memory",
            label="Project Feedback Memory",
            status="disabled",
            summary="项目反馈记忆未启用。",
        )
    memory_path = (
        pathlib.Path(config.memory_path)
        if config.memory_path
        else pathlib.Path(workspace_path) / ".citation-agent" / "memory.sqlite3"
    )
    if memory_path.exists():
        return HealthCheckItem(
            name="memory",
            label="Project Feedback Memory",
            status="ok",
            summary="项目反馈记忆数据库存在。",
            details=[f"path={memory_path}"],
            actions=["可通过 EasyCite: 清空项目记忆 重置反馈。"],
        )
    return HealthCheckItem(
        name="memory",
        label="Project Feedback Memory",
        status="warning",
        summary="项目反馈记忆已启用，首次接受或拒绝候选后会创建数据库。",
        details=[f"path={memory_path}"],
    )


def module_available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False
