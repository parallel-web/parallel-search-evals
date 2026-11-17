"""Measure every production module with the frozen quality gates."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import measure as analyzer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
versions = {name: importlib.metadata.version(name) for name in analyzer.PINS}
if versions != analyzer.PINS or sys.version_info[:2] != (3, 12):
    raise RuntimeError("Measurement requires Python 3.12 and the frozen tool versions")
analyzer_hash = hashlib.sha256(Path(analyzer.__file__).read_bytes()).hexdigest()
if analyzer_hash != "e82485313dcb3480df4486e37549626fa06d3bbed038e00446a710e63a3893f9":
    raise RuntimeError("The frozen analyzer changed")
scopes = {
    label: analyzer.measure(root, folder)
    for label, folder in (
        ("parallel_search_evals", "src/parallel_search_evals"),
        ("tests", "tests"),
        ("scripts", ".github"),
    )
}
report = {
    "production": "src/parallel_search_evals/**/*.py including all initializers",
    "tools": versions,
    "python": platform.python_version(),
    "analyzer_sha256": analyzer_hash,
    "scopes": scopes,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            key: scopes["parallel_search_evals"][key]
            for key in (
                "mean_complexity",
                "mean_mi",
                "ruff_count",
                "module_count",
                "function_count",
                "lines",
                "decision_excess",
                "worst_function",
                "lowest_module",
                "parser_diagnostics",
                "gates_pass",
            )
        },
        indent=2,
    )
)

if not scopes["parallel_search_evals"]["gates_pass"]:
    raise SystemExit(1)
