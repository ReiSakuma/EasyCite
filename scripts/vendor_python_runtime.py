#!/usr/bin/env python3
"""Copy required Python runtime packages into bundled/libs for VSIX packaging."""
from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path


REQUIRED_PACKAGES = (
    "annotated_types",
    "pydantic",
    "pydantic_core",
    "typing_inspection",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    site_packages = Path(sysconfig.get_paths()["purelib"])
    target_root = repo_root / "bundled" / "libs"
    target_root.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied: list[str] = []
    for package in REQUIRED_PACKAGES:
        package_path = site_packages / package
        if not package_path.exists():
            missing.append(package)
            continue
        copy_tree(package_path, target_root / package)
        copied.append(package)

        for dist_info in site_packages.glob(f"{package.replace('_', '-')}*.dist-info"):
            copy_tree(dist_info, target_root / dist_info.name)
        for dist_info in site_packages.glob(f"{package}*.dist-info"):
            copy_tree(dist_info, target_root / dist_info.name)

    if missing:
        print(
            "Missing required package(s): "
            + ", ".join(missing)
            + f"\nInstall them into {sys.executable} before packaging.",
            file=sys.stderr,
        )
        return 1

    print(
        "Vendored Python runtime packages into bundled/libs: "
        + ", ".join(copied)
    )
    return 0


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
