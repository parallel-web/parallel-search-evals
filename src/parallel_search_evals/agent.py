"""Answer questions using native model history and an optional Parallel tool loop.

Only question text enters this component; reference answers stay with the grader.
Each answer owns its history, tool-call budget and overall deadline. Model and
Parallel HTTP clients are shared across answers and closed by the run owner.
Finalization disables tools and records the final-answer instruction in both
native history and the separate instructions field.
"""

from __future__ import annotations

import asyncio

from parallel_search_evals import config
from parallel_search_evals.config import (
    CALL_SECONDS,
    CLOSED_BOOK_INSTRUCTIONS,
    FINAL_ANSWER_INSTRUCTIONS,
    GENERATION_SECONDS,
    MAX_TOOL_CALLS,
    RESEARCH_INSTRUCTIONS,
    RESEARCH_TURNS,
)
from parallel_search_evals.models import ModelHistory, ModelReply, ModelSettings, open_model
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher, ToolSpec
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import TOOLS, ToolChoice


class AnswerGenerator:
    def __init__(
        self,
        research_mode: str,
        model_settings: ModelSettings,
        max_turns: int = RESEARCH_TURNS + 1,
        *,
        filter_benchmarks: bool = config.FILTER_BENCHMARKS,
    ) -> None:
        self.model_client = open_model(model_settings)
        self.max_turns = max_turns
        self.research_mode = research_mode
        self.research_client: ParallelClient | None = None
        self.tool_specs: tuple[ToolSpec, ...] = ()
        if research_mode.startswith("parallel-"):
            self.research_client = ParallelClient(
                mode=research_mode.removeprefix("parallel-"),
                model=self.model_client.model,
                filter_benchmarks=filter_benchmarks,
            )
            self.tool_specs = TOOLS

    async def answer(self, question_text: str) -> ModelHistory:
        history: ModelHistory = self.model_client.new_history()
        history.system = CLOSED_BOOK_INSTRUCTIONS if self.research_mode == "baseline" else RESEARCH_INSTRUCTIONS
        history.append_question(question_text)
        tools = ToolDispatcher(self.tool_specs, max_calls=MAX_TOOL_CALLS, client=self.research_client)
        async with asyncio.timeout(GENERATION_SECONDS):
            for step in range(self.max_turns):
                finalizing = step == self.max_turns - 1 or tools.remaining_calls == 0
                tools.tool_choice = ToolChoice.NONE if finalizing or not tools.actions else ToolChoice.AUTO
                if finalizing:
                    history.system = RESEARCH_INSTRUCTIONS + "\n\n" + FINAL_ANSWER_INSTRUCTIONS
                async with asyncio.timeout(CALL_SECONDS):
                    output = await self.model_client.generate(history, tools)
                history.accept_reply(output)
                if not output.tool_calls:
                    if not output.text.strip():
                        raise NativeResponseError("Model returned no current final answer")
                    return history
                await self._apply_tool_calls(history, tools, output, step)
        raise NativeResponseError("Model did not produce a final answer")

    async def _apply_tool_calls(
        self, history: ModelHistory, tools: ToolDispatcher, output: ModelReply, step: int
    ) -> None:
        ids = [call.id for call in output.tool_calls]
        if tools.tool_choice == ToolChoice.NONE or len(set(ids)) != len(ids):
            raise NativeResponseError("Repeated or unavailable native tool call")
        results = await tools.execute(output.tool_calls)
        final_next = step == self.max_turns - 2 or tools.remaining_calls == 0
        # Record finalization in both native history and the separate instructions.
        history.append_tool_outputs(results, FINAL_ANSWER_INSTRUCTIONS if final_next else "")
