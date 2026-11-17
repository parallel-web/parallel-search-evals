# Evaluation methodology

Compare the same exact source questions, prompts, graders, tool limits and native model settings. Provider routes, aliases and reasoning names are not interchangeable. Save the source revision and run configuration with reported results.

## Generation

Baseline is closed-book. Parallel turbo, fast, basic and advanced modes expose local Search and Extract tools. Research permits 20 executed tool calls and 25 research turns, followed when needed by one tools-disabled final call. Each tool call receives a matching result or error. The exact baseline/research/finalization prompts are source constants in `config.py`.

Generation has an absolute 5400-second overall deadline and 3600-second (one-hour) per-model-call deadline. Each tool operation has an absolute 1800-second deadline and 600-second HTTP operation timeout. Official judging has an absolute 1800-second deadline, with 30-second connection and 120-second idle-read timeouts. Generation uses native nonstreaming SDK calls; the judge streams. No automatic retry, provider fallback or `MAX_TOKENS` recovery is provided. Native Google `CONTINUATION` tokens may resume a logical generation for at most 32 physical hops within the same 3600-second call and 5400-second overall deadlines. The 32-hop bound accommodates the public Gemini 4 output ceiling across 32,768-token nonstreaming segments on Vertex `v1beta1`. Logical generation/tool limits stay 26/20; a malformed, missing or repeated token or exhausted hop allowance fails without grading partial text.

Search and Extract use Parallel's service defaults for result counts and response size. Search explicitly sends the selected benchmark mode and required search queries. With the default benchmark source filter, local processing removes Search and Extract result/error entries whose URLs match known benchmark source prefixes. Other entries, excerpts, warnings and native metadata are preserved. An Extract request containing any blocked URL rejects the whole batch before HTTP and returns a tool error; the model can submit another request with allowed URLs. This uses the existing tool-call budget. With `filter_benchmarks=False`, the original full response reaches the model unchanged. The harness retains its research turn limits, tool-call limits and network timeouts.

## Questions and leakage

DSQA uses the pinned CSV; HLE uses a user-provisioned pinned text-only JSONL export. Selection takes the first `limit` eligible source-order rows. References go only to official graders, never generation prompts. The default retrieval filter uses a finite list of known benchmark source URL paths. Search filtering happens locally after the service responds; it does not prevent the service from retrieving those sources. The filter does not compare question text or discover unknown mirrors, and benchmark material on other URLs may still reach the model. Do not claim contamination-free evaluation. Record the filter setting and stored source list when comparing runs; `filter_benchmarks=False` explicitly opts out. See [dataset terms](third-party/README.md#dataset-terms).

## Grading and reporting

Both suites use `gpt-6-sol`, medium reasoning and 16,384 output tokens. DSQA uses the pinned official starter-notebook prompt and question-level fractional F1. HLE uses the pinned upstream `JUDGE_PROMPT` and strict correctness. This is the harness judge model, not a reproduction of the original upstream judge model. Official schema strictness remains false for DSQA and true for HLE. [Prompt provenance](../src/parallel_search_evals/benchmarks/assets/official-sources.json) and [notices](third-party/README.md#software-provenance) are packaged.

Malformed grading or generation is a failed terminal, distinct from a valid zero score. All planned tasks, including failures, remain in the denominator. Report task count, failures, exact source/model/mode/settings and the limitations of a three-question smoke. Such a sample is not a reproduction of published full-benchmark scores.
