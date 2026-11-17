"""Empty official selections reject before clients or run output construction."""

import hashlib

import pytest

from parallel_search_evals import models, run_eval


@pytest.mark.parametrize("suite", ["dsqa", "hle"])
def test_validly_pinned_empty_cohort_never_creates_clients_or_summary(tmp_path, monkeypatch, suite):
    source = tmp_path / "empty-source"
    source.write_text("problem,problem_category,answer,answer_type\n" if suite == "dsqa" else "")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")

    def forbidden(**kwargs):
        raise AssertionError("Empty selected cohort reached client construction")

    monkeypatch.setattr(models, "AsyncOpenAI", forbidden)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="no selected"):
        run_eval.main("baseline", suite, "gpt-fixture", source, output_dir=output, source_sha256=digest, synthetic=True)
    assert not output.exists() and not (tmp_path / "results").exists()
