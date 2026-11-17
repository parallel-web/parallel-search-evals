"""Offline negative cases for repository content, documentation and lock audit boundaries."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / ".github/scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_document_examples_parse_actual_cli_without_executing(tmp_path, monkeypatch):
    checker = script("check_docs")

    def forbidden(*args, **kwargs):
        raise AssertionError("Documentation check must not execute the evaluator")

    monkeypatch.setattr(checker.run_eval, "run_evaluation", forbidden)
    (tmp_path / "README.md").write_text(
        "```sh\nuv run --frozen python -m parallel_search_evals.run_eval search_engine=baseline suite=dsqa model=gpt-fixture source=/synthetic/input.csv\n```\n"
    )
    result = checker.check(tmp_path)
    assert result == {"local_links": 0, "parsed_commands": 1, "failures": []}


@pytest.mark.parametrize("argument", ["unknown_option=True", "max_workers=invalid"])
def test_document_unknown_arguments_and_bad_types_fail(tmp_path, argument):
    checker = script("check_docs")
    (tmp_path / "README.md").write_text(
        "[missing](absent.md)\n```sh\nuv run --frozen python -m parallel_search_evals.run_eval search_engine=baseline suite=dsqa model=gpt-fixture source=/synthetic/input.csv "
        + argument
        + "\n```\n"
    )
    result = checker.check(tmp_path)
    assert result["parsed_commands"] == 0 and len(result["failures"]) == 2


def test_lock_audit_includes_all_registry_versions_and_rejects_other_sources():
    checker = script("check_dependencies")
    local = {"name": "parallel-search-evals", "version": "0.1.0", "source": {"editable": "."}}
    public = {"name": "fixture", "version": "1.2.3", "source": {"registry": "https://pypi.org/simple"}}
    assert checker.locked_requirements({"package": [local, public, public]}) == ["fixture==1.2.3"]
    with pytest.raises(ValueError, match="unsupported dependency source"):
        checker.locked_requirements({"package": [dict(public, source={"git": "https://example.invalid/repo"})]})


def test_content_scanner_reports_rule_and_path_without_secret_value():
    checker = script("check_public")
    secret = b"sk-" + b"X" * 40
    findings = checker.inspect("synthetic.txt", secret)
    assert findings == [{"file": "synthetic.txt", "rule": "credential"}]
    assert secret.decode() not in str(findings)
