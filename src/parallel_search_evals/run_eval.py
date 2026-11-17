"""Preserve the public chz command and validate its invocation.

This module owns CLI defaults, credential-presence checks and mode selection.
Run configuration, client lifetimes and terminal reuse belong to run.py.
Keep main's callable signature compatible with the documented
python -m parallel_search_evals.run_eval command and its existing arguments.
"""

import asyncio
import os
from pathlib import Path

import chz

from parallel_search_evals import config
from parallel_search_evals.config import configure_logging
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.run import run_evaluation


def main(
    search_engine: str,
    suite: str,
    model: str,
    source: Path,
    output_dir: Path | None = None,
    limit: int | None = config.DEFAULT_QUESTION_LIMIT,
    source_sha256: str | None = None,
    synthetic: bool = False,
    max_tokens: int | None = None,
    provider: str | None = None,
    openrouter_backends: tuple[str, ...] = (),
    base_url: str | None = None,
    api: str = "responses",
    api_key_env: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    effort: str = "default",
    max_workers: int = config.DEFAULT_MAX_WORKERS,
    filter_benchmarks: bool = config.FILTER_BENCHMARKS,
    judge_assets: Path | None = None,
) -> None:
    configure_logging()
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if search_engine not in {"baseline", *(f"parallel-{mode}" for mode in config.PARALLEL_MODES)}:
        raise ValueError("Select baseline or a supported Parallel mode")
    required = {"OPENAI_API_KEY"}
    if search_engine != "baseline":
        required.add("PARALLEL_API_KEY")
    if not all(os.environ.get(key, "").strip() for key in required):
        raise ValueError("Missing required API credentials")
    settings = ModelSettings.from_selection(
        {
            "model": model,
            "provider": provider,
            "max_tokens": max_tokens,
            "openrouter_backends": openrouter_backends,
            "base_url": base_url,
            "api": api,
            "api_key_env": api_key_env,
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
            "effort": effort,
        }
    )
    run_name = f"{search_engine}-{model.replace('/', '_')}_{suite}"
    output_dir = output_dir or Path("runs") / run_name
    asyncio.run(
        run_evaluation(
            search_engine,
            settings,
            suite,
            output_dir,
            max_workers,
            source,
            limit,
            source_sha256,
            synthetic,
            filter_benchmarks,
            judge_assets,
        )
    )


if __name__ == "__main__":
    chz.entrypoint(main)
