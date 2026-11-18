# Parallel Search Evals

[![CI](https://github.com/parallel-web/parallel-search-evals/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/parallel-web/parallel-search-evals/actions/workflows/ci.yml?query=branch%3Amain)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Evaluate language models with [Parallel](https://parallel.ai/) Search and Extract
tools against a closed-book baseline. Supports native OpenAI, Anthropic and Google
APIs, OpenRouter, Unbiased, and OpenAI-compatible endpoints.
Search and Extract use the official [Parallel Python SDK](https://github.com/parallel-web/parallel-sdk-python).

Complete [setup](#setup), obtain the benchmark data locally, then run one model
on one benchmark with Parallel search:

```sh
# Smoke test: three questions, paid model, search and grading calls.
uv run --frozen python -m parallel_search_evals.run_eval \
  provider=anthropic model=claude-sonnet-5 suite=dsqa search_engine=parallel-fast \
  source=/absolute/path/DSQA-full.csv limit=3 max_workers=2 \
  output_dir=runs/dsqa-smoke

# Full benchmark: all 900 DSQA questions.
uv run --frozen python -m parallel_search_evals.run_eval \
  provider=anthropic model=claude-sonnet-5 suite=dsqa search_engine=parallel-fast \
  source=/absolute/path/DSQA-full.csv limit=None max_workers=2 \
  output_dir=runs/dsqa-full

# Show the actual CLI options without making provider requests.
uv run --frozen python -m parallel_search_evals.run_eval --help
```

## Results

[View the Search Capability Leaderboard](https://parallel.ai/leaderboard).

## Benchmarks

| Benchmark | Questions | Description | Sources |
| --- | ---: | --- | --- |
| `dsqa` | 900 | DeepSearchQA: multi-step web research across 17 fields, testing complete answer sets. | [Paper](https://arxiv.org/abs/2601.20975), [benchmark](https://www.kaggle.com/benchmarks/google/dsqa) |
| `hle` | 2,158[^benchmark-counts] | Humanity’s Last Exam: text-only expert-written academic questions across dozens of subjects. | [Paper](https://arxiv.org/abs/2501.14249), [dataset](https://huggingface.co/datasets/cais/hle) |

[^benchmark-counts]: Counts describe the available source questions before selection
    limits. HLE’s published dataset has 2,500 questions; the text-only count excludes
    the 342 image questions [documented for the pinned revision](https://github.com/google-deepmind/gbench/blob/main/docs/evals/humanitys_last_exam.md#dataset).
    This implementation selects only rows with `image=""`.

DSQA and HLE use pinned official grading prompts with **GPT-6 Sol** as the judge.
See [methodology](docs/methodology.md) for scoring and
[dataset terms](docs/third-party/README.md#dataset-terms) for access and redistribution requirements.
No benchmark corpus is bundled or automatically downloaded.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/) for the frozen quality toolchain.

```sh
git clone https://github.com/parallel-web/parallel-search-evals.git
cd parallel-search-evals
uv sync --frozen --python 3.12
```

Configure credentials privately in the process environment for the evaluated model
and the OpenAI grader. Search-enabled runs also require `PARALLEL_API_KEY`.
Obtain the benchmark data under its upstream terms and pass a local `source` path;
the harness does not authenticate to dataset hosts or accept terms for you.

## Run an evaluation

Each invocation evaluates **one model × one benchmark × one mode**. Use `parallel-fast`, `parallel-basic`, `parallel-advanced`
or `parallel-turbo` for Parallel Search and Extract, or `baseline` for closed-book
generation. Run comparisons separately with matched questions and settings, then
`uv run --frozen python -m parallel_search_evals.compare` on the two run directories.

The CLI defaults to the first three eligible source-order questions and concurrency
ten. Set `limit=N` for a sample, `limit=None` for all eligible rows, and
`max_workers=1` for a sequential smoke run. Selection is deterministic; it does not
sample or shuffle questions.

Models beginning with `gpt` select native OpenAI Responses and models beginning
with `claude` select native Anthropic Messages unless `provider` is explicit.
Other routes require an explicit provider:

| `provider` | Native API | Selection and authentication |
| --- | --- | --- |
| `openai` | Responses | `OPENAI_API_KEY` |
| `anthropic` | Messages | `ANTHROPIC_API_KEY` |
| `openrouter` | Responses | `OPENROUTER_API_KEY`, optional `openrouter_backends`; explicit pins disable provider fallback |
| `unbiased` | Responses | `UNBIASED_API_KEY`, explicit `max_tokens` |
| `compatible` | Responses or Chat Completions | Explicit `base_url`, `api=responses` or `api=chat`, `max_tokens`; `COMPATIBLE_API_KEY` or an `api_key_env` variable name |
| `gemini` | Google Developer contents/parts | `GOOGLE_API_KEY` |
| `vertex` | Vertex contents/parts | Explicit `vertex_project`, `vertex_location`, application default credentials |

The official grader requires `OPENAI_API_KEY` on every route. All search-enabled
routes use the same local Parallel tools. Use each provider’s exact available model
ID and supported settings. The harness does not retry with another provider or
model. Omitted OpenRouter backend pins delegate provider selection to OpenRouter.


## Results and resume

Each run writes `config.json` and one terminal `{task_id}.json` per question.
Completed records contain the original datum, native conversation and official
grade; failed records contain the datum and failure. The CLI logs progress and
summary results; a summary is saved next to the run in `results/{run_name}.json`.
All planned questions, including failures, remain in the score denominator.

To resume, rerun the same command with the same output directory. Valid terminal
results, including failures, are reused. Missing or malformed records rerun the
whole task; partial answers are not checkpoints. A different configuration is
rejected before provider requests. Use one writer per directory.

Run records may contain restricted benchmark content and sensitive provider data;
keep them private. See [operations](docs/operations.md) for format compatibility,
terminal reuse and result interpretation. 

## Citation

Record the exact source revision and saved configuration when reporting results.

```bibtex
@misc{parallel_search_evals_2026,
  author = {{Parallel Web Systems}},
  title = {Parallel Search Evals},
  year = {2026},
  url = {https://github.com/parallel-web/parallel-search-evals}
}
```

## License

Framework code is [MIT licensed](LICENSE).
[Dataset terms](docs/third-party/README.md#dataset-terms) and
[third-party notices](docs/third-party/README.md#software-provenance) apply separately. See
[contributing](.github/CONTRIBUTING.md) to extend the harness and
[security](.github/SECURITY.md) to report a vulnerability.
