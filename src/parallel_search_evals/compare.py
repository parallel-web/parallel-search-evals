"""Compare two finished run directories question by question.

Reads saved configs and terminal records only. Opens no clients, needs no
credentials, and never prints question text, answers, grade text or history.
"""

from __future__ import annotations

import math
import operator
import sys
from pathlib import Path
from typing import Any

import chz
import orjson

from parallel_search_evals.results import AnswerRecord, EvaluationSummary, ScoredAnswer, read_answer
from parallel_search_evals.storage import regular_file

IDENTITY = (
    "suite",
    "source_sha256",
    "source",
    "synthetic",
    "judge_model",
    "filter_benchmarks",
    "benchmark_sources",
)
MCNEMAR = "exact two-sided McNemar (binomial)"
CELLS = {
    (True, True): "both_correct",
    (True, False): "only_base_correct",
    (False, True): "only_candidate_correct",
    (False, False): "neither",
}


class CompareError(Exception):
    """Unmatched or unreadable run directories."""


def mcnemar_p_value(only_base: int, only_candidate: int) -> float:
    """Return the exact two-sided McNemar p-value.

    Discordant pairs are scored with math.comb on a binomial(n, 1/2) tail.
    """
    n = sum((only_base, only_candidate))
    k = min(only_base, only_candidate)
    if n:
        tail = sum(math.comb(n, i) for i in range(sum((k, 1))))
        return min(1.0, math.ldexp(sum((tail, tail)), operator.neg(n)))
    return 1.0


def load_config(directory: Path, side: str) -> dict[str, Any]:
    """Load a run config.json mapping.

    Missing directories and unreadable files each raise one clear message.
    """
    path = directory.joinpath("config.json")
    if directory.is_dir():
        try:
            with regular_file(path) as stream:
                payload = orjson.loads(stream.read())
        except (FileNotFoundError, OSError, ValueError) as error:
            raise CompareError(f"Unreadable config ({side}): {path}") from error
        if isinstance(payload, dict):
            return payload
        raise CompareError(f"Unreadable config ({side}): {path}")
    raise CompareError(f"Missing directory ({side}): {directory}")


def load_records(directory: Path) -> dict[str, AnswerRecord]:
    """Index readable terminal records by task-id filename stem.

    Unreadable JSON files are omitted from the matched task set.
    """
    records: dict[str, AnswerRecord] = {}
    for path in sorted(path for path in directory.glob("*.json") if operator.ne(path.name, "config.json")):
        record = read_answer(path)
        if record:
            records[path.stem] = record
    return records


def mismatched_identity(base: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    """Return the first identity field that differs.

    judge_assets is compared only when at least one config records it.
    """
    names = list(IDENTITY)
    if {"judge_assets"}.intersection(base).union({"judge_assets"}.intersection(candidate)):
        names.append("judge_assets")
    return next((name for name in names if operator.ne(base.get(name), candidate.get(name))), None)


def require_tasks(base: dict[str, AnswerRecord], candidate: dict[str, AnswerRecord]) -> None:
    """Reject when readable task-id sets differ.

    Messages include the missing count and the first few identifiers.
    """
    clauses = []
    for label, ids in (
        ("base", sorted(set(candidate).difference(base))),
        ("candidate", sorted(set(base).difference(candidate))),
    ):
        if ids:
            shown = ", ".join(ids[:8])
            if operator.gt(len(ids), 8):
                shown = f"{shown}, ..."
            clauses.append(f"{len(ids)} missing from {label} ({shown})")
    if clauses:
        raise CompareError(f"Unmatched task IDs: {'; '.join(clauses)}")


def _correct(record: AnswerRecord) -> bool:
    """Count a completed grade as correct; failures are incorrect."""
    if isinstance(record, ScoredAnswer):
        return record.grader_result.correct
    return False


def classify(base: dict[str, AnswerRecord], candidate: dict[str, AnswerRecord]) -> dict[str, list[str]]:
    """Bucket each shared task ID into a paired-correctness cell.

    Failures count as incorrect, matching EvaluationSummary.from_records.
    """
    cells: dict[str, list[str]] = {name: [] for name in CELLS.values()}
    for task_id in sorted(base):
        cells[CELLS[(_correct(base[task_id]), _correct(candidate[task_id]))]].append(task_id)
    return cells


def _side(summary: EvaluationSummary, config: dict[str, Any]) -> dict[str, Any]:
    """Public counters for one run.

    Restricted question, answer, grade and history text are omitted.
    """
    return {
        "score": summary.score,
        "total_correct": summary.total_correct,
        "failed_samples": summary.failed_samples,
        "total_samples": summary.total_samples,
        "search_engine": config.get("search_engine"),
        "synthetic": config.get("synthetic"),
    }


def compare_runs(base: Path, candidate: Path) -> dict[str, Any]:
    """Match identity and task IDs, then report paired scores and McNemar.

    search_engine and model may differ; that difference is the comparison.
    """
    base_config, candidate_config = load_config(base, "base"), load_config(candidate, "candidate")
    field = mismatched_identity(base_config, candidate_config)
    if field:
        raise CompareError(f"Unmatched configuration: {field}")
    base_records, candidate_records = load_records(base), load_records(candidate)
    require_tasks(base_records, candidate_records)
    cells = classify(base_records, candidate_records)
    base_summary = EvaluationSummary.from_records(list(base_records.values()))
    candidate_summary = EvaluationSummary.from_records(list(candidate_records.values()))
    only_base, only_candidate = len(cells["only_base_correct"]), len(cells["only_candidate_correct"])
    return {
        "differences": [
            key for key in ("search_engine", "model") if operator.ne(base_config.get(key), candidate_config.get(key))
        ],
        "base": _side(base_summary, base_config),
        "candidate": _side(candidate_summary, candidate_config),
        "paired": {name: len(cells[name]) for name in CELLS.values()},
        "failures": {"base": base_summary.failed_samples, "candidate": candidate_summary.failed_samples},
        "net_change": {
            "score": operator.sub(candidate_summary.score, base_summary.score),
            "correct": operator.sub(candidate_summary.total_correct, base_summary.total_correct),
        },
        "mcnemar": {
            "method": MCNEMAR,
            "p_value": mcnemar_p_value(only_base, only_candidate),
            "discordant": sum((only_base, only_candidate)),
        },
        "ids": {
            "only_base_correct": cells["only_base_correct"],
            "only_candidate_correct": cells["only_candidate_correct"],
        },
    }


def _line(name: str, side: dict[str, Any]) -> str:
    """One human-readable summary line for a run directory.

    Includes synthetic so demo output can label fixture runs.
    """
    return (
        f"{name}: score={side['score']:.6f} correct={side['total_correct']}/"
        f"{side['total_samples']} failed={side['failed_samples']} "
        f"search_engine={side['search_engine']} synthetic={side['synthetic']}"
    )


def render_text(payload: dict[str, Any], include_ids: bool) -> str:
    """Format the human report without restricted record fields.

    Discordant task IDs are listed only when include_ids is true.
    """
    paired, change, test = payload["paired"], payload["net_change"], payload["mcnemar"]
    diffs = ", ".join(payload["differences"])
    lines = [
        _line("base", payload["base"]),
        _line("candidate", payload["candidate"]),
        f"differing settings: {diffs if diffs else 'none'}",
        "paired (failures counted as incorrect):",
        *_paired_lines(paired),
        (
            f"failures (counted as incorrect): base={payload['failures']['base']} "
            f"candidate={payload['failures']['candidate']}"
        ),
        f"net change: score={change['score']:+.6f} correct={change['correct']:+d}",
        f"{MCNEMAR} p-value: {test['p_value']:.10g}",
    ]
    if include_ids:
        lines.extend(_id_lines(payload["ids"]))
    lines.append("")
    return "\n".join(lines)


def _json_text(payload: dict[str, Any]) -> str:
    """Encode a comparison object as indented JSON with a trailing newline."""
    return f"{orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()}\n"


def _paired_lines(paired: dict[str, int]) -> list[str]:
    """Format paired-cell counts in a stable cell order."""
    return [f"  {name}: {paired[name]}" for name in CELLS.values()]


def _id_lines(ids: dict[str, list[str]]) -> list[str]:
    """Format discordant task identifiers without record text."""
    lines: list[str] = []
    for cell, values in ids.items():
        lines.append(f"{cell} ids:")
        lines.extend(f"  {task_id}" for task_id in values)
    return lines


def main(base: Path, candidate: Path, json: bool = False, ids: bool = False) -> None:
    """Compare two run directories using flat name=value CLI arguments.

    json=true prints a machine-readable object; ids=true lists discordant IDs.
    """
    try:
        payload = compare_runs(base, candidate)
    except CompareError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
    if json:
        dumped = dict(payload)
        if ids:
            sys.stdout.write(_json_text(dumped))
            return
        dumped.pop("ids")
        sys.stdout.write(_json_text(dumped))
        return
    sys.stdout.write(render_text(payload, ids))


if operator.eq(__name__, "__main__"):
    chz.entrypoint(main)
