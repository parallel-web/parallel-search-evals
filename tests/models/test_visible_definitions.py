"""ToolDispatcher controls whether its registered definitions are visible on a model turn."""

import pytest

from parallel_search_evals.models.anthropic_messages import messages_tool_schemas
from parallel_search_evals.models.openai_responses import responses_tool_schemas
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import EXTRACT, SEARCH, ToolChoice


@pytest.mark.parametrize("render", [responses_tool_schemas, messages_tool_schemas])
def test_disabled_tools_hide_definitions_without_losing_registration(render):
    tools = ToolDispatcher([SEARCH, EXTRACT])
    active = render(tools.available)
    assert len(active) == 2
    tools.tool_choice = ToolChoice.NONE
    assert render(tools.available) == []
    assert len(tools.actions) == 2
    tools.tool_choice = ToolChoice.AUTO
    assert render(tools.available) == active
