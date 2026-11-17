# Contributing

Use Python 3.12 and uv; install development dependencies with `uv sync --frozen --python 3.12`.

Install the credential-scanning Git hook once per clone:

```sh
uvx --from pre-commit==4.7.0 pre-commit install --config .github/pre-commit-config.yml
```

The pinned Gitleaks hook checks staged files before a commit. It does not
verify candidate credentials over the network or print their values. Run the same
checks against the checkout and its commit history with:

```sh
uvx --from pre-commit==4.7.0 pre-commit run --config .github/pre-commit-config.yml --hook-stage manual --all-files
```

CI runs this check independently of local hook installation. Review findings rather
than adding broad exclusions or a baseline that suppresses existing credentials.

```sh
uv run --frozen ruff check src tests .github/quality/check.py .github/scripts
uv run --frozen ruff format --check src tests .github/quality/check.py .github/scripts
uv run --frozen mypy --strict --no-incremental src/parallel_search_evals
uv run --frozen pytest -q
uv run --frozen python .github/scripts/check_docs.py
uv run --frozen python .github/quality/check.py --output metrics.json
uv run --frozen python .github/scripts/check_public.py
uv build
uv run --frozen python .github/scripts/check_package.py
```

The supported entry point is the CLI. Python modules are implementation interfaces, not a stable SDK. Import types directly from defining modules. Keep unvalidated SDK/JSON controls at validated boundaries and preserve opaque native replay separately. Strict typing does not prove runtime validation or that every SDK argument is checked.

Tests use synthetic questions and deterministic transports at actual native SDK boundaries, with sockets/DNS denied and no paid requests. Shared fixture helpers own construction; scenario tests own assertions. Cover IDs/order, unknown/null controls, signed native state, malformed envelopes, local tool errors, deadlines, sibling cancellation, client closure and terminal reload. Do not silently change prompts, routes, budgets or scoring during structural work.

## Execution and extension

`run_eval.main` resolves immutable `ModelSettings`. `run.py` owns construction, client lifetimes and evaluation scheduling. Its runner reads terminal records or answers, grades and stores each question before computing the full-denominator summary. Partial answers are not checkpoints.

- Model: add explicit selection/auth settings in `models/settings.py` and its model factory, then implement the native protocol with request, continuation, error and history tests. Preserve native IDs and opaque reasoning state.
- Benchmark: register its loader, rendering, scoring and assets through `benchmarks/base.py`. Keep suite behavior in its benchmark module and shared judge transport in `benchmarks/judge.py`. Pin provenance and document dataset terms; restricted questions and references stay outside the repository.
- Tool: define schemas and validated inputs in `tools/schemas.py`, execute them through `ToolDispatcher`, and keep HTTP admission in `tools/parallel_client.py`. Exercise multiple calls, errors and finalization at the tool limit.
- Execution: `agent.py` owns answer generation, `config.py` owns runtime settings and generation instructions, `runtime.py` owns persisted failure identities and task joining, and `results.py` owns terminal records and summaries. `storage.py` owns secure filesystem operations.

The frozen analyzer measures every production Python module, including initializers. Required whole-production gates are mean CC≤3, mean MI≥70 and isolated Ruff≤20; normal CI Ruff must be clean. Do not modify the frozen analyzer, omit source or move production logic into tests/scripts to satisfy gates. Parser diagnostics and generic-function body counts must remain disclosed.

## Review and live checks

Describe the problem, resulting behavior and exact validation in PRs. Keep private traces, credentials, reference answers and dataset rows out of issues, PRs and attachments. Use [security](SECURITY.md) for vulnerability reports. CI runs offline quality/package/security-content checks and Ubuntu/macOS tests, with a separate locked dependency audit. It never dispatches paid smoke runs.

Any paid validation is a separate, explicitly bounded private operation with exact model/provider/backend/caps and approved local data. Record physical outcomes and safe evidence; a provider error or SSH wrapper exit alone does not prove integration success. Do not substitute model aliases or silently retry.
