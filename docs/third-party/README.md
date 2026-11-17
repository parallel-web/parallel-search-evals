# Third-party licenses and dataset terms

Framework code is [MIT licensed](../../LICENSE). Full upstream license texts are retained in this directory. Dataset terms apply separately from code and prompt licenses.

## Software provenance

Generation/grading methodology and official prompt/schema assets reproduce the pinned Parallel Search Evals reference at `e54ebda82130ca725405aa42e04e64fdbcf35101`. Its unmodified MIT notice is retained in [the root LICENSE](../../LICENSE). Extraction provenance and hashes are recorded in `src/parallel_search_evals/benchmarks/assets/official-sources.json`. The lean scope and explicit differences are documented separately; this is not a claim of full main parity.

The DSQA grading prompt and partial example preserve version 1 of the official starter notebook by Google DeepMind / Andrew Ming Wang (`https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code`). Its complete Apache-2.0 text is retained without modification in [google-deepsearchqa-Apache-2.0.txt](google-deepsearchqa-Apache-2.0.txt). Original placeholders and extraction hashes are recorded alongside `src/parallel_search_evals/benchmarks/assets/dsqa-judge.user.txt`; this attribution does not grant rights to benchmark data.

DSQA reads the public `google/deepsearchqa` dataset at revision `b2623f8653065c2672de6d941fc5434cd652376c` (Apache-2.0). No dataset rows are bundled. The framework's MIT license does not license third-party data. The pinned source checksum is recorded in the loader and verification evidence.

The HLE prompt preserves the upstream `JUDGE_PROMPT` from `centerforaisafety/hle` revision `22ed3074b1e7b134bcbc09028d0ba320839b0655`. Its unmodified MIT notice is packaged at `src/parallel_search_evals/benchmarks/assets/hle-LICENSE.txt`; provenance remains in `official-sources.json`.

HLE input is a user-provisioned local export verified against a supplied digest. The reference methodology identifies gated `cais/hle` revision `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`; a local checksum alone does not establish that upstream revision. Verify provenance, obtain access and accept upstream terms separately. The dataset card restricts sharing, re-uploading and distribution of rows. Neither the framework nor prompt license grants dataset access or redistribution rights. No restricted HLE rows are included; fixtures are synthetic.

## Google DeepMind / DeepSearchQA

`src/parallel_search_evals/benchmarks/assets/dsqa-judge.user.txt` preserves the grading prompt and partial example from version 1 of the [official starter notebook](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code). Attribution: Google DeepMind / Andrew Ming Wang. Its complete Apache-2.0 license is retained without modification in [licenses/google-deepsearchqa-Apache-2.0.txt](google-deepsearchqa-Apache-2.0.txt). Original placeholders and source hashes are recorded in [official-sources.json](../../src/parallel_search_evals/benchmarks/assets/official-sources.json). This attribution does not grant rights to benchmark data.

## centerforaisafety/hle

`src/parallel_search_evals/benchmarks/assets/hle-judge.user.txt` preserves the upstream `JUDGE_PROMPT` at commit `22ed3074b1e7b134bcbc09028d0ba320839b0655`. Its unmodified MIT notice remains packaged as `src/parallel_search_evals/benchmarks/assets/hle-LICENSE.txt`. Main's complete upstream notice is also retained in [licenses/centerforaisafety-hle-MIT.txt](centerforaisafety-hle-MIT.txt).

## openai/simple-evals

Main's conservative attribution and complete upstream MIT notice are retained in [licenses/openai-simple-evals-MIT.txt](openai-simple-evals-MIT.txt), pinned to `652c89d0ca9df547706735883097e9537d40dc47`. This preserves upstream notice continuity and does not claim that the absent binary judge is packaged here or grant benchmark-data rights.

# Dataset terms

This repository contains adapters and synthetic tests. It does not distribute benchmark questions, reference answers, gated rows or proprietary benchmark content. The framework's MIT licenses do not license third-party data.

## DeepSearchQA (DSQA)

- Source: [`google/deepsearchqa`](https://huggingface.co/datasets/google/deepsearchqa)
- Immutable revision: `b2623f8653065c2672de6d941fc5434cd652376c`
- File: `DSQA-full.csv`, 900 source questions
- SHA-256: `25d48dcf7efa872e5467032e8b8eedf38d301f59a252d0da95cda584baa78396`
- Upstream license: Apache-2.0

Obtain the public file yourself and pass its local path as `source`. This implementation does not download data. The loader verifies the pinned checksum before parsing, selects the first `limit` source-order rows, and preserves reference strings verbatim. `limit=None` selects all rows. An explicitly synthetic CSV needs `synthetic=True` and its own `source_sha256`; synthetic tests are integration checks, not benchmark scores.

## Humanity's Last Exam (HLE)

- Source: [`cais/hle`](https://huggingface.co/datasets/cais/hle)
- Immutable metadata revision: `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`
- Evaluation code: [`centerforaisafety/hle`](https://github.com/centerforaisafety/hle)
- Access: gated on Hugging Face

The dataset metadata carries an MIT label, while its card asks users not to publicly share, re-upload or distribute it. Obtain access and download the data under those upstream conditions yourself. The harness does not authenticate to Hugging Face or accept terms. Supply a local upstream-shaped JSONL export and its SHA-256 using `source_sha256`. Only rows with `image=""` are selected. Parquet and normalized aliases are not supported.

Run artifacts contain selected questions and reference answers and must not be committed, uploaded or distributed. The HLE prompt license does not grant dataset access or redistribution rights. Keep all run directories private unless their contents have been reviewed and sharing is permitted.

## Local evidence and identity

Terminal task files contain the original datum and, for completed tasks, the native conversation and official grade. Configuration is stored separately in `config.json`. These files are not content-free. Use a new directory when the source or settings change and only one writer per directory. Source-order selection and terminal reuse differ from main's seeded cohort and checkpoint machinery; see [operations](../operations.md).
