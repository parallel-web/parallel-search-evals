import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from parallel_search_evals import run as evaluation_run
from parallel_search_evals import run_eval
from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.results import FailedAnswer, write_answer
from tests.run.helpers import FixtureRunner


class Closed:
    async def aclose(self):
        pass

    async def close(self):
        pass


class Suite(FixtureRunner):
    def __init__(self, output_dir, calls):
        super().__init__(output_dir, 1)
        self.questions = [Question(id="0", problem="Synthetic question", answer="Synthetic reference")]
        self.calls = calls
        self.generator = SimpleNamespace(model_client=Closed(), research_client=None)
        self.grader = SimpleNamespace(client=Closed())

    async def _evaluate_question(self, datum):
        self.calls.append(datum.id)
        return FailedAnswer(datum=datum, failure="Synthetic terminal")


def terminal(message="Synthetic terminal"):
    return FailedAnswer(datum=Question(id="0", problem="Q", answer="A"), failure=message)


async def evaluate(output, monkeypatch, calls):
    monkeypatch.setattr(evaluation_run, "build_evaluation", lambda *args, **kwargs: Suite(kwargs["output_dir"], calls))
    await run_eval.run_evaluation(
        "baseline", ModelSettings.from_selection({"model": "gpt-fixture"}), "hle", output, 1, output.parent / "input"
    )


async def test_new_durable_outputs_are_private_under_public_umask(tmp_path, monkeypatch):
    output = tmp_path / "new-parent" / "run"
    old = os.umask(0o022)
    try:
        await evaluate(output, monkeypatch, [])
    finally:
        os.umask(old)
    for directory in (output.parent, output, output.parent / "results"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in (
        output / "config.json",
        output / "0.json",
        output.parent / "results/run.json",
        output.parent / ".run.lock",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("operation", ["replace", "fsync"])
def test_failed_atomic_write_preserves_old_terminal_and_cleans_temporary(tmp_path, monkeypatch, operation):
    path = tmp_path / "0.json"
    write_answer(terminal(), path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("Synthetic storage failure")

    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError, match="Synthetic storage"):
        write_answer(terminal("Replacement"), path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0.json"]


def test_atomic_destination_symlink_does_not_overwrite_external_file(tmp_path):
    external = tmp_path / "external"
    external.write_bytes(b"outside sentinel")
    path = tmp_path / "0.json"
    path.symlink_to(external)
    write_answer(terminal(), path)
    assert external.read_bytes() == b"outside sentinel"
    assert not path.is_symlink() and json.loads(path.read_bytes())["status"] == "failed"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("alias", ["direct", "relative", "symlink-parent"])
async def test_other_process_lock_rejects_before_client_factory_then_releases(tmp_path, monkeypatch, alias):
    output = tmp_path / "run"
    output.mkdir()
    if alias == "relative":
        (tmp_path / "existing").mkdir()
        output = tmp_path / "existing/../run"
    elif alias == "symlink-parent":
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        output = tmp_path / "alias/run"
    lock_path = output.resolve().parent / ".run.lock"
    code = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.read()"
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout.readline().strip() == "ready"

    def forbidden(*args, **kwargs):
        pytest.fail("Contended run reached remote client factory")

    monkeypatch.setattr(evaluation_run, "build_evaluation", forbidden)
    try:
        with pytest.raises(RuntimeError, match="writer"):
            await run_eval.run_evaluation(
                "baseline", ModelSettings.from_selection({"model": "gpt-fixture"}), "hle", output, 1, tmp_path / "input"
            )
        assert not (output / "config.json").exists()
    finally:
        holder.communicate(input="", timeout=10)
    calls = []
    await evaluate(output, monkeypatch, calls)
    before = {p: p.read_bytes() for p in output.rglob("*.json")}
    await evaluate(output, monkeypatch, calls)
    assert calls == ["0"]
    assert before == {p: p.read_bytes() for p in output.rglob("*.json")}


@pytest.mark.parametrize("link", ["directory", "lock", "config"])
async def test_symlink_admission_fails_before_clients(tmp_path, monkeypatch, link):
    output = tmp_path / "run"
    outside = tmp_path / "outside"
    if link == "directory":
        outside.mkdir()
        output.symlink_to(outside, target_is_directory=True)
    else:
        output.mkdir()
        outside.write_bytes(b"outside sentinel")
        target = tmp_path / ".run.lock" if link == "lock" else output / "config.json"
        target.symlink_to(outside)

    def forbidden(*args, **kwargs):
        pytest.fail("Symlink admission reached remote client factory")

    monkeypatch.setattr(evaluation_run, "build_evaluation", forbidden)
    with pytest.raises(OSError):
        await run_eval.run_evaluation(
            "baseline", ModelSettings.from_selection({"model": "gpt-fixture"}), "hle", output, 1, tmp_path / "input"
        )
    assert list(outside.iterdir()) == [] if link == "directory" else outside.read_bytes() == b"outside sentinel"


async def test_sibling_runs_and_summary_files_do_not_share_a_lock(tmp_path, monkeypatch):
    calls = []
    await asyncio.gather(evaluate(tmp_path / "one", monkeypatch, calls), evaluate(tmp_path / "two", monkeypatch, calls))
    assert calls == ["0", "0"]
    assert json.loads((tmp_path / "results/one.json").read_bytes())["total_samples"] == 1
    assert json.loads((tmp_path / "results/two.json").read_bytes())["total_samples"] == 1


@pytest.mark.parametrize("leaf", ["config", "lock", "task"])
def test_fifo_run_files_fail_without_blocking_or_client_creation(tmp_path, leaf):
    output = tmp_path / "run"
    output.mkdir()
    path = tmp_path / ".run.lock" if leaf == "lock" else output / ("config.json" if leaf == "config" else "0.json")
    os.mkfifo(path)
    code = "\nimport asyncio,socket,sys\nfrom pathlib import Path\ndef denied(*args,**kwargs):raise AssertionError('Network forbidden')\nsocket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=denied\nfrom parallel_search_evals import run_eval\nfrom parallel_search_evals import run as evaluation_run\nfrom parallel_search_evals.models.settings import ModelSettings\nfrom parallel_search_evals.results import read_answer\ndef forbidden(*args,**kwargs):raise AssertionError('FIFO reached client factory')\nevaluation_run.build_evaluation=forbidden\noutput=Path(sys.argv[1])\ntry:\n    if sys.argv[2]=='task':read_answer(output/'0.json')\n    else:asyncio.run(run_eval.run_evaluation('baseline',ModelSettings.from_selection({'model':'gpt-fixture'}),'hle',output,1,output/'input'))\nexcept OSError:pass\nelse:raise AssertionError('FIFO was admitted')\n"
    result = subprocess.run(
        [sys.executable, "-c", code, str(output), leaf],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("previous_judge", [None, "gpt-5.6-sol"])
async def test_judge_identity_blocks_old_runs_before_clients_and_current_runs_reuse(
    tmp_path, monkeypatch, previous_judge
):
    output = tmp_path / "run"
    calls = []
    await evaluate(output, monkeypatch, calls)
    config_path = output / "config.json"
    config = json.loads(config_path.read_bytes())
    assert config["judge_model"] == "gpt-6-sol"
    terminal_before = (output / "0.json").read_bytes()
    await evaluate(output, monkeypatch, calls)
    assert calls == ["0"]
    assert (output / "0.json").read_bytes() == terminal_before
    if previous_judge is None:
        del config["judge_model"]
    else:
        config["judge_model"] = previous_judge
    config_path.write_text(json.dumps(config))
    previous_config = config_path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Old judge configuration reached provider setup")

    monkeypatch.setattr(evaluation_run, "build_evaluation", forbidden)
    with pytest.raises(ValueError, match="different"):
        await run_eval.run_evaluation(
            "baseline",
            ModelSettings.from_selection({"model": "gpt-fixture"}),
            "hle",
            output,
            1,
            output.parent / "input",
        )
    assert config_path.read_bytes() == previous_config
    assert (output / "0.json").read_bytes() == terminal_before
