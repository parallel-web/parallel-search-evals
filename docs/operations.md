# Operations and terminal reuse

Run one model × one suite × one mode per invocation. For comparisons, invoke baseline and search separately with matched source-order questions and model settings. All model/tool/judge calls are paid; pytest uses offline synthetic fixtures.

`config.json` stores ordinary applied run settings and source identity. Each `{task_id}.json` stores either a completed datum/conversation/official grade or a failed datum/failure. The summary is written beside the run as `results/{run_name}.json`; failures remain in its planned denominator. Progress and summary logging are emitted by the CLI. No standalone summary command or CSV export is provided.

Rerunning identical arguments against the same directory reuses valid terminal results, including failures. Missing or malformed task files rerun the entire task. Partial answers are not checkpoints. A mismatched configuration is rejected before provider requests.

Run configurations record the selected model settings, research mode, benchmark, source identity, `judge_model`, `filter_benchmarks` and `benchmark_sources`. The filter is enabled by default; `benchmark_sources` always records the known URL prefix list; `filter_benchmarks` records whether it applies. Reuse an output directory only with the same configuration. Changing the filter flag or source list requires a fresh directory. Older configurations without these fields are rejected, including when opting out.

A canonical exclusive process lock admits one writer per run, including path aliases. New output directories and files use private permissions (0700 and 0600); result files are replaced atomically after syncing. There is no raw-request ledger or interrupted-phase recovery. A process/storage/programming error can abort the run rather than become benchmark evidence. Keep a copy of private evidence before manual recovery work; do not edit completed records to alter outcomes.

Raw native conversations and original references can contain restricted material or sensitive data. The product does not promise comprehensive provider-body redaction. Keep the whole run directory private, inspect summaries before sharing, and follow [security](../.github/SECURITY.md) and [dataset terms](third-party/README.md#dataset-terms). Git ignore patterns are convenience, not a security boundary.

Parallel Search/Extract requests carry `User-Agent: parallel-search-evals/<installed-version>` for integration attribution. This header contains only the package identity and version.
