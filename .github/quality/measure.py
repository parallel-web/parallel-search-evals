"""Apply measurement-contract.md unchanged to each enumerated Python source scope."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import lizard
from radon.metrics import mi_visit
from radon.raw import analyze

PINS = {"lizard": "1.17.31", "radon": "6.0.1", "ruff": "0.15.22"}


def measure(root, scope):
    directories = [root / scope]
    if scope == "scripts":
        directories.append(root / ".github" / "scripts")
    entries = [path for directory in directories for path in (directory, *directory.rglob("*"))]
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"Symlink in measured scope: {path}")
    paths = sorted(path for path in entries if path.suffix == ".py" and path.is_file())
    modules = []
    functions = []
    parser_diagnostics = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symlink in measured source: {path}")
        source = path.read_text()
        raw = analyze(source)
        tree = ast.parse(source)
        measured = lizard.analyze_file(str(path)).function_list
        declared = Counter(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        observed = Counter(function.name.rsplit(".", 1)[-1] for function in measured)
        if observed != declared:
            parser_diagnostics.append(
                {
                    "path": str(path.relative_to(root)),
                    "python_only": dict(declared - observed),
                    "lizard_only": dict(observed - declared),
                }
            )
        modules.append(
            {
                "path": str(path.relative_to(root)),
                "physical_lines": len(source.splitlines()),
                "sloc": raw.sloc,
                "lloc": raw.lloc,
                "comments": raw.comments,
                "blank": raw.blank,
                "classes": sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree)),
                "mi": mi_visit(source, multi=True),
                "mi_without_multiline_comments": mi_visit(source, multi=False),
            }
        )
        for function in measured:
            functions.append(
                {
                    "path": str(path.relative_to(root)),
                    "line": function.start_line,
                    "name": function.name,
                    "complexity": function.cyclomatic_complexity,
                }
            )
    diagnostics = []
    if paths:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--no-preview",
                "--target-version",
                "py312",
                "--line-length",
                "100",
                "--ignore-noqa",
                "--no-respect-gitignore",
                "--select",
                "C90,PLR,SIM,RET,ARG",
                "--output-format",
                "json",
                *map(str, paths),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        diagnostics = json.loads(result.stdout)
        for diagnostic in diagnostics:
            diagnostic["filename"] = str(Path(diagnostic["filename"]).relative_to(root))
    mean_cc = statistics.mean(f["complexity"] for f in functions) if functions else None
    mean_mi = statistics.mean(m["mi"] for m in modules) if modules else None
    return {
        "parser_diagnostics": parser_diagnostics,
        "module_count": len(modules),
        "function_count": len(functions),
        "classes": sum(m["classes"] for m in modules),
        "lines": {
            key: sum(m[key] for m in modules)
            for key in (
                "physical_lines",
                "sloc",
                "lloc",
                "comments",
                "blank",
            )
        },
        "mean_complexity": mean_cc,
        "decision_excess": sum(f["complexity"] - 1 for f in functions),
        "worst_function": max(functions, key=lambda f: f["complexity"], default=None),
        "mean_mi": mean_mi,
        "lowest_module": min(modules, key=lambda m: m["mi"], default=None),
        "ruff_count": len(diagnostics),
        "ruff_by_code": dict(Counter(d["code"] for d in diagnostics)),
        "gates_pass": (
            mean_cc is not None
            and mean_cc <= 3
            and mean_mi is not None
            and mean_mi >= 70
            and len(diagnostics) <= 20
        ),
        "modules": modules,
        "functions": functions,
        "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual = {name: importlib.metadata.version(name) for name in PINS}
    if actual != PINS or sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Frozen toolchain mismatch: {actual}, {sys.version}")
    root = args.root.resolve()
    lock = root / "uv.lock"
    project = tomllib.loads((root / "pyproject.toml").read_text())
    report = {
        "root": ".",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tools": actual,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "settings": {
            "production": "src/**/*.py",
            "separate_scopes": ["tests/**/*.py", "scripts/**/*.py", ".github/scripts/**/*.py"],
            "lizard": "function-weighted mean; default analysis; no extensions",
            "radon": "unweighted module mean; multi=True; multi=False also reported",
            "ruff": "isolated stable py312 line100 C90,PLR,SIM,RET,ARG ignore-noqa McCabe10",
            "thresholds": {"max_mean_cc": 3, "max_ruff": 20, "min_mean_mi": 70},
        },
        "runtime_dependencies": project["project"].get("dependencies", []),
        "scopes": {scope: measure(root, scope) for scope in ("src", "tests", "scripts")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for scope, value in report["scopes"].items():
        print(
            scope,
            json.dumps(
                {
                    key: value[key]
                    for key in (
                        "mean_complexity",
                        "mean_mi",
                        "ruff_count",
                        "module_count",
                        "function_count",
                        "lines",
                        "worst_function",
                        "lowest_module",
                        "decision_excess",
                        "gates_pass",
                    )
                }
            ),
        )
    production = report["scopes"]["src"]
    return 0 if production["gates_pass"] and not production["parser_diagnostics"] else 1


if __name__ == "__main__":
    sys.exit(main())
