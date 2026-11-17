# Configuration

The supported entry point is `python -m parallel_search_evals.run_eval`, using flat `name=value` arguments. Python modules are implementation interfaces, not a stable SDK. There is no TOML configuration layer or automatic model discovery.

## Providers and credentials

See the [README provider table](../README.md#run-an-evaluation) for all eight native API routes and required environment-variable names. The official grader always needs `OPENAI_API_KEY`; search-enabled modes also need `PARALLEL_API_KEY`. Compatible routes require an explicit endpoint, API protocol and output cap. OpenRouter backend tags are optional. Omit them to let OpenRouter select its provider; when supplied, tags must be unique and nonblank and are sent in exact order with provider fallback disabled. Vertex requires explicit project/location and externally provisioned application default credentials. The harness does not configure cloud credentials or accept dataset terms.

Compatible endpoints require HTTPS, except HTTP on localhost or a literal loopback address. `api_key_env` accepts uppercase environment-variable names (`[A-Z_][A-Z0-9_]*`), not credential values. Harness-owned model and judge clients reject redirects.

Set credentials privately in the process environment. Never place values in CLI arguments, configuration examples, committed files or issue reports. Only selected routes load their relevant credentials. Native OpenAI generation and judging retain intended native organization/project environment settings; nonnative OpenAI-compatible routes omit ambient tenant headers. Explicit Anthropic API-key selection omits ambient bearer authentication.

## Defaults and overrides

The CLI defaults to three source-order questions and concurrency ten; `max_workers=1` is useful for a smoke run. OpenAI, Anthropic and OpenRouter generation caps fall back to 128,000; Google Developer/Vertex fall back to 65,536. These known model/route combinations have specific defaults:

| Route and model | Default output cap | Public limit source |
| --- | ---: | --- |
| Vertex `gemini-4-carbon`, `gemini-4-hydrogen`, `gemini-4-argon` | 1,048,576 | Google [Carbon](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/4-carbon), [Hydrogen](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/4-hydrogen), [Argon](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/4-argon) |
| OpenRouter `moonshotai/kimi-k3.1-pro`, exactly `openrouter_backends=river` | 1,048,575 | [OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/moonshotai/kimi-k3.1-pro/endpoints) |
| OpenRouter `moonshotai/kimi-k3.1-pro`, model default including Moonshot | 943,718 | [OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/moonshotai/kimi-k3.1-pro/endpoints) |
| OpenRouter `tencent/hy4` | 65,536 | [OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/tencent/hy4/endpoints) |

Google 4 has a unified input/output budget: thinking, tool calls and final text share the 1,048,576-token context with input. The configured cap is an output ceiling; available output is limited by remaining context. Vertex uses `v1beta1`, which supports long-decode continuation; older Gemini models retain their smaller defaults. These overrides are exact model/route defaults, not automatic catalog discovery.

Kimi backend limits differ. The River override applies after backend validation and only to the exact single-backend selection; Moonshot and mixed selections retain the model default. River reports unknown quantization, while the Moonshot `moonshotai/mxfp4` route reports MXFP4. Record the selected backend when comparing results, and choose an explicit supported cap for other endpoints.

Compatible and Unbiased routes require an explicit positive cap. Explicit `max_tokens` overrides any default. A provider's current model or selected backend may support a smaller limit: select the exact supported model/cap/effort together. Anthropic Messages retains its 128000 interactive limit; a larger batch-only beta limit does not apply to this API. The harness does not retry with another model or provider. Explicit OpenRouter backend pins disable provider fallback; omitted pins delegate provider selection to OpenRouter.

`effort=default` resolves provider defaults, `effort=none` omits effort. Native OpenAI effort suffixes and Anthropic's `-thinking` suffix resolve only on their respective native routes; explicit effort overrides an OpenAI suffix. Applied physical settings are persisted. Settings irrelevant to a route are ignored by transport but still recorded conservatively in run identity.

Responses requests omit the optional `tool_choice=auto` parameter and use the API default when tools are available. Explicit `none` and `required` controls are preserved.

Model generations allow one hour (3,600 seconds) per logical call, including all Google continuation requests, within the 5,400-second overall generation deadline. SDK request timeouts match the one-hour call limit.

`filter_benchmarks=True` is the default. Search and Extract responses are filtered locally against a finite list of known benchmark source URL prefixes; an Extract batch containing any blocked URL is rejected before HTTP. Set `filter_benchmarks=False` to disable both checks and forward the original full responses. This setting does not add server-side Search constraints or change response-size defaults. See [methodology](methodology.md#questions-and-leakage) for its limits.

Use a new output directory whenever source, model, route, mode, prompts, timeout defaults, benchmark source policy or applied settings change. See [operations](operations.md) for terminal reuse and [methodology](methodology.md) for fixed grading and budgets. Run `uv run --frozen python -m parallel_search_evals.run_eval --help` to inspect actual CLI parameters without making provider requests.

## Local exports and external judges

Use `suite=local` for a locally licensed JSONL export with string `id`, `problem`
and `answer` fields. Other string fields are retained as metadata. Supply its
SHA-256 and a `judge_assets` directory containing four files:

- `judge.system.txt`: system instructions; may be empty.
- `judge.user.txt`: a nonempty user prompt with `{question_json}`, `{reference_json}`,
  `{answer_json}` and optional `{answer_type_json}` substitutions. Values are JSON-escaped.
- `judge-schema.json`: a valid JSON object schema requiring `extracted_final_answer`,
  `reasoning`, `correct` (`"yes"` or `"no"`) and integer `confidence` from 0 to 100.
- `LICENSE`: the nonempty license notices applicable to those judging assets.

```sh
uv run --frozen python -m parallel_search_evals.run_eval \
  suite=local provider=openai model=gpt-6-sol search_engine=parallel-fast \
  source=/path/to/questions.jsonl source_sha256=YOUR_SOURCE_SHA256 \
  judge_assets=/path/to/judge-assets output_dir=runs/local-smoke limit=3
```

Only the supplied templates and schema enter judge requests; license text stays
local. All four files are fingerprinted before model clients open. Changing any
asset requires a fresh output directory. Local grading independently enforces four strict binary judgment fields, even
when the supplied schema is permissive; it uses the shared judge model and deadlines. DSQA and HLE keep their pinned official assets
and reject `judge_assets` overrides. No local corpus or judge bundle is packaged.
