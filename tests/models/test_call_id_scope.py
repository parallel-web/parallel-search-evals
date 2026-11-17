"""Call identities are validated per response before dispatch."""

import pytest

import parallel_search_evals.agent as generation
from parallel_search_evals.models import ResponsesHistory
from parallel_search_evals.models.base import ModelReply
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools.schemas import ToolInvocation


@pytest.mark.parametrize("steps,identities", [(2, ["duplicate", "duplicate"]), (1, ["disabled"])])
async def test_invalid_current_calls_fail_before_any_tool_dispatch(monkeypatch, steps, identities):
    class Model:
        model = "fixture"

        def new_history(self):
            return ResponsesHistory(assistant_text="")

        async def generate(self, history, tools):
            return ModelReply(
                texts=[],
                tool_calls=[
                    ToolInvocation(id=identity, name="search_web", input={"objective": "q", "search_queries": ["q"]})
                    for identity in identities
                ],
                native=[],
            )

    async def forbidden_dispatch(*args, **kwargs):
        pytest.fail("Invalid call reached the tool HTTP boundary")

    monkeypatch.setenv("PARALLEL_API_KEY", "fixture-not-a-secret")
    monkeypatch.setattr(generation, "open_model", lambda model_settings: Model())
    monkeypatch.setattr(generation.ParallelClient, "request", forbidden_dispatch)
    agent = generation.AnswerGenerator(
        "parallel-basic", ModelSettings.from_selection({"model": "gpt-fixture", "max_tokens": 64}), max_turns=steps
    )
    try:
        with pytest.raises(NativeResponseError, match="Repeated or unavailable"):
            await agent.answer("question")
    finally:
        await agent.research_client.sdk.close()
