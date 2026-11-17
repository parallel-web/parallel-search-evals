"""Invalid external prompt formatting is rejected before paid model construction."""

import chz
import pytest

from parallel_search_evals import models, run_eval
from tests.test_local_cli import local_inputs


@pytest.mark.parametrize("template,error", [("Unknown {unsupported}", KeyError), ("Unclosed {", ValueError)])
def test_bad_local_prompt_stops_before_clients(tmp_path, monkeypatch, template, error):
    assets, output, argv = local_inputs(tmp_path)
    (assets / "judge.user.txt").write_text(template)
    clients = []

    def forbidden(**kwargs):
        clients.append(kwargs)
        pytest.fail("Invalid local prompt reached model construction")

    monkeypatch.setattr(models, "AsyncOpenAI", forbidden)
    with pytest.raises(error):
        chz.entrypoint(run_eval.main, argv=[*argv, "search_engine=baseline"])
    assert not clients and not output.exists()
