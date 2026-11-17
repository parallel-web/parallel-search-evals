"""Runtime defaults shared by the CLI, models, tools and benchmark judge.

Keep effective values here so transport and selection code share one policy.
Resolved model settings and the judge model are saved by the runner; changing
these defaults can therefore affect which saved runs remain compatible.
Prompt text below is sent verbatim, while logging defaults apply process-wide.
This module has no imports from execution code and can be used during setup.
"""

import logging
from typing import Final

CALL_SECONDS = 3600
GENERATION_SECONDS = 5400
JUDGE_SECONDS = 1800
TOOL_SECONDS = 1800
PARALLEL_REQUEST_SECONDS = 600
CONNECT_SECONDS = 30
JUDGE_IDLE_SECONDS = 120
MAX_TOOL_CALLS = 20
RESEARCH_TURNS = 25
MAX_CONTINUATION_HOPS = 32


# CLI defaults also apply when the runner is assembled directly.
DEFAULT_QUESTION_LIMIT = 3
DEFAULT_MAX_WORKERS = 10
FILTER_BENCHMARKS = True

# Judge selections are recorded in the saved run configuration.
JUDGE_MODEL = "gpt-6-sol"
JUDGE_MAX_OUTPUT_TOKENS = 16384
JUDGE_REASONING_EFFORT: Final = "medium"

# SDK retry counts express total attempts differently across vendors.
SDK_MAX_RETRIES = 0
PROMPT_CACHE_KEY = "parallel-search-evals"
GOOGLE_ATTEMPTS = 1
ANTHROPIC_THINKING_BUDGET = 1024

PARALLEL_BASE_URL = "https://api.parallel.ai"
PARALLEL_DEFAULT_MODE = "basic"
PARALLEL_MODES = {"turbo", "fast", "basic", "advanced"}

ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "unbiased": "https://api.unbiased.ai/v1",
}

KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "unbiased": "UNBIASED_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "compatible": "COMPATIBLE_API_KEY",
}

DEFAULTS = {
    "openai": (128000, "high"),
    "anthropic": (128000, "max"),
    "openrouter": (128000, "high"),
    "unbiased": (None, None),
    "compatible": (None, None),
    "gemini": (65536, "high"),
    "vertex": (65536, "high"),
}

MODEL_MAX_TOKENS = {
    ("vertex", "gemini-4-carbon"): 1048576,
    ("vertex", "gemini-4-hydrogen"): 1048576,
    ("vertex", "gemini-4-argon"): 1048576,
    ("openrouter", "moonshotai/kimi-k3.1-pro"): 943718,
    ("openrouter", "tencent/hy4"): 65536,
}

MODEL_BACKEND_MAX_TOKENS: dict[tuple[str, str, tuple[str, ...]], int] = {
    ("openrouter", "moonshotai/kimi-k3.1-pro", ("river",)): 1048575,
}


RESEARCH_INSTRUCTIONS = (
    "Answer the question. Use the search and extraction tools to research the question and "
    "ground your answer in retrieved evidence. Choose useful queries and follow up as needed. "
    "Give a clear final answer supported by the evidence; if you do not know, say so."
)

CLOSED_BOOK_INSTRUCTIONS = (
    "Answer the question using your own knowledge. Search and extraction tools "
    "are unavailable. Give your best answer; if you do not know, say so."
)

FINAL_ANSWER_INSTRUCTIONS = (
    "Research is finished. Tools are now unavailable. Give your final answer using the evidence "
    "already collected. If the evidence is insufficient, say so."
)


SDK_LOGGERS = [
    "asyncio",
    "httpx",
    "openai",
    "anthropic",
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s:%(lineno)d %(message)s")
    for name in SDK_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
