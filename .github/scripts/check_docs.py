"""Offline local-link and actual CLI argument/config checks; never execute paid commands."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import re
import shlex
from functools import wraps
from pathlib import Path

import chz

from parallel_search_evals import run_eval
from parallel_search_evals.models.settings import ModelSettings

ROOT = Path(__file__).resolve().parents[2]


def shell_command(line: str) -> list[str]:
    words = shlex.split(line, comments=True)
    if words[:2] != ["uv", "run"]:
        return words
    index = 2
    while index < len(words):
        if words[index] in {"--env-file", "--project", "--directory", "--python"}:
            index += 2
        elif words[index].startswith("-"):
            index += 1
        else:
            return words[index:]
    return []


def cli_arguments(line: str) -> list[str] | None:
    words = shell_command(line)
    if len(words) >= 3 and words[0] == "python" and words[1:3] == ["-m", "parallel_search_evals.run_eval"]:
        return words[3:]
    return None


def check(root: Path) -> dict:
    checked_links = commands = 0
    failures = []
    documents = sorted(root.glob("*.md"))
    documents += sorted((root / ".github").glob("*.md"))
    documents += sorted((root / "docs").rglob("*.md"))
    documents += sorted((root / ".agents" / "skills").rglob("*.md"))
    for document in documents:
        text = document.read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            local = target.split("#")[0]
            checked_links += 1
            if not (document.parent / local).exists():
                failures.append(f"{document.name}: missing local link {local}")
        for block in re.findall(r"```(?:sh|bash|shell)\n(.*?)```", text, re.S):
            for line in block.replace("\\\n", " ").splitlines():
                words = shell_command(line)
                if (
                    len(words) >= 2
                    and re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(words[0]).name)
                    and words[1].endswith(".py")
                    and not (root / words[1]).is_file()
                ):
                    failures.append(f"{document.name}: missing local script {words[1]}")
                args = cli_arguments(line)
                if args is None:
                    continue
                # Parsing validates real flags even for commands that require licensed data.
                try:
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):

                        @wraps(run_eval.main)
                        def validate(**values):
                            fields = ModelSettings.model_fields
                            defaults = inspect.signature(run_eval.main).parameters
                            selected = {
                                name: values.get(name, defaults[name].default) for name in fields if name in defaults
                            }
                            ModelSettings.from_selection(selected).validate_selected_route()

                        chz.Blueprint(validate).make_from_argv(args)
                except chz.blueprint.EntrypointHelpException:
                    commands += 1
                    continue
                except (
                    ValueError,
                    chz.blueprint.ExtraneousBlueprintArg,
                    chz.blueprint.InvalidBlueprintArg,
                    chz.blueprint.MissingBlueprintArg,
                ):
                    failures.append(f"{document.name}: invalid CLI arguments")
                    continue
                commands += 1
    return {"local_links": checked_links, "parsed_commands": commands, "failures": failures}


def main() -> None:
    result = check(ROOT)
    print(json.dumps(result))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
