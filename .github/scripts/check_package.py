"""Check the exact distributions and load their installed resources outside the source tree."""

import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    wheels = list((ROOT / "dist").glob("*.whl"))
    sources = list((ROOT / "dist").glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "build one wheel and source archive first"
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel = {name: archive.read(name) for name in archive.namelist()}
    with tarfile.open(sources[0]) as archive:
        source = {}
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                source[member.name.split("/", 1)[-1]] = stream.read()
    policy_paths = [ROOT / "uv.lock", ROOT / ".gitignore"]
    policy_paths += [
        path
        for folder in (".github", ".agents", "docs")
        for path in (ROOT / folder).rglob("*")
        if path.is_file() and path.suffix in {".md", ".yml", ".py"}
    ]
    for path in policy_paths:
        name = str(path.relative_to(ROOT))
        assert source[name] == (ROOT / name).read_bytes(), f"missing or altered {name}"
    notices = (
        "LICENSE",
        *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs/third-party").iterdir() if p.is_file()),
    )
    for name in notices:
        data = (ROOT / name).read_bytes()
        assert source[name] == data
        assert any(key.endswith("/licenses/" + name) and value == data for key, value in wheel.items()), name
    for path in sorted((ROOT / "src/parallel_search_evals").rglob("*.py")):
        name = str(path.relative_to(ROOT))
        assert wheel[str(path.relative_to(ROOT / "src"))] == path.read_bytes() == source[name], name
    expected_modules = {str(path.relative_to(ROOT)) for path in (ROOT / "src/parallel_search_evals").rglob("*.py")}
    wheel_modules = {
        "src/" + name for name in wheel if name.startswith("parallel_search_evals/") and name.endswith(".py")
    }
    source_modules = {name for name in source if name.startswith("src/parallel_search_evals/") and name.endswith(".py")}
    assert wheel_modules == source_modules == expected_modules, "Distribution module inventory differs from source"
    assets = sorted((ROOT / "src/parallel_search_evals/benchmarks/assets").glob("*"))
    assets += sorted((ROOT / "src/parallel_search_evals/tools/assets").glob("*"))
    for path in assets:
        if path.is_file() and path.suffix != ".py":
            name = str(path.relative_to(ROOT))
            assert wheel[str(path.relative_to(ROOT / "src"))] == path.read_bytes() == source[name], name
    with tempfile.TemporaryDirectory(prefix="eval-wheel-") as directory:
        work = Path(directory)
        installed = work / "installed"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.endswith("API_KEY") and key not in {"HF_TOKEN", "PYTHONPATH"}
        }
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--target",
                str(installed),
                str(wheels[0]),
            ],
            cwd=work,
            env=env,
            check=True,
        )
        env["PYTHONPATH"] = str(installed)
        help_result = subprocess.run(
            [sys.executable, "-m", "parallel_search_evals.run_eval", "--help"],
            cwd=work,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert help_result.returncode == 1 and "Entry point: main" in help_result.stdout
        assert "search_engine" in help_result.stdout and "Traceback" not in help_result.stderr
        code = "from pathlib import Path; import parallel_search_evals.run_eval as entry; from importlib.resources import files; assert Path(entry.__file__).resolve().is_relative_to(Path('installed').resolve()); assert files('parallel_search_evals.benchmarks').joinpath('assets').joinpath('dsqa-judge.user.txt').read_text(); assert files('parallel_search_evals.benchmarks').joinpath('assets').joinpath('hle-judge.user.txt').read_text()"
        subprocess.run([sys.executable, "-c", code], cwd=work, env=env, check=True)
    print(f"Installed wheel outside checkout, {len(assets)} resource entries and distribution notices passed.")


if __name__ == "__main__":
    main()
