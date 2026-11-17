"""Audit every registry package/version in uv.lock, including other platforms and extras."""

from __future__ import annotations

import subprocess
import tempfile
import tomllib
from pathlib import Path


def locked_requirements(lock: dict) -> list[str]:
    requirements = set()
    for package in lock["package"]:
        source = package["source"]
        if source == {"editable": "."} and package["name"] == "parallel-search-evals":
            continue
        if source != {"registry": "https://pypi.org/simple"}:
            raise ValueError(f"unsupported dependency source: {package['name']}")
        requirements.add(f"{package['name']}=={package['version']}")
    if not requirements:
        raise ValueError("no locked registry dependencies to audit")
    return sorted(requirements)


def main() -> None:
    requirements = locked_requirements(tomllib.loads(Path("uv.lock").read_text()))
    with tempfile.TemporaryDirectory(prefix="eval-dependency-audit-") as directory:
        path = Path(directory) / "requirements.txt"
        path.write_text("\n".join(requirements) + "\n")
        print(f"Auditing {len(requirements)} locked package/version pairs", flush=True)
        subprocess.run(
            [
                "uvx",
                "--from",
                "pip-audit==2.10.1",
                "pip-audit",
                "--strict",
                "--disable-pip",
                "--no-deps",
                "--progress-spinner",
                "off",
                "-r",
                str(path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
