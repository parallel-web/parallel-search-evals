"""External binary-judge assets and generic normalized local question rendering.

Judge files are supplied locally, not downloaded or packaged. Their original
bytes identify the grading policy for result reuse. Templates and schema enter
judge requests; the required license is retained and fingerprinted locally.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

from jsonschema import Draft202012Validator
from pydantic import ConfigDict, TypeAdapter

from parallel_search_evals.benchmarks.base import Question
from parallel_search_evals.storage import regular_file

ASSET_NAMES = ("judge.system.txt", "judge.user.txt", "judge-schema.json", "LICENSE")
JUDGE_SCHEMA = TypeAdapter(dict[str, Any], config=ConfigDict(hide_input_in_errors=True))


class JudgeAssets(NamedTuple):
    """Keep one decoded snapshot for both configuration and judge requests.

    Bundled judges need no external fingerprints. Local snapshots hash all four
    files; license text is excluded from the decoded request fields.
    """

    system: str
    template: str
    schema: dict[str, Any]
    sha256: dict[str, str]


def load_judge_assets(benchmark: str, directory: Path | None) -> JudgeAssets | None:
    """Require explicit assets for local runs while keeping official judges fixed."""
    if directory is None:
        if benchmark == "local":
            raise ValueError("suite=local requires judge_assets")
        return None
    if benchmark != "local":
        raise ValueError("judge_assets is only supported for suite=local")
    return read_judge_assets(directory)


def read_judge_assets(directory: Path) -> JudgeAssets:
    """Admit one complete asset snapshot before any model client is constructed.

    Hash the original bytes so whitespace and license changes invalidate resume.
    Decode that same snapshot for requests rather than reopening mutable files.
    """
    contents = {}
    for name in ASSET_NAMES:
        with regular_file(directory / name) as stream:
            contents[name] = stream.read()
    if not contents["judge.user.txt"].strip() or not contents["LICENSE"].strip():
        raise ValueError("Local judge prompt and license must be nonempty")
    schema = JUDGE_SCHEMA.validate_json(contents["judge-schema.json"], strict=True)
    Draft202012Validator.check_schema(schema)
    template = contents["judge.user.txt"].decode()
    template.format(question_json="", reference_json="", answer_json="", answer_type_json="")
    return JudgeAssets(
        contents["judge.system.txt"].decode(),
        template,
        schema,
        {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    )


def render(template: str, question: Question, answer: str) -> str:
    """Quote supplied question, reference and response text as JSON string values."""
    return template.format(
        question_json=json.dumps(question.problem, ensure_ascii=False),
        reference_json=json.dumps(question.answer, ensure_ascii=False),
        answer_json=json.dumps(answer, ensure_ascii=False),
        answer_type_json=json.dumps(question.metadata.get("answer_type"), ensure_ascii=False),
    )
