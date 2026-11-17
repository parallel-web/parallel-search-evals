"""Trusted pinned action definitions render native dictionaries without sharing nested assets."""

from copy import deepcopy
from dataclasses import replace

import pytest

from parallel_search_evals.models.anthropic_messages import messages_tool_schemas
from parallel_search_evals.models.openai_responses import responses_tool_schemas
from parallel_search_evals.tools.schemas import EXTRACT, SCHEMAS, SEARCH


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("definition,index", [(SEARCH, 0), (EXTRACT, 1)])
def test_schema_boundary_returns_exact_native_dictionary(definition, index, provider):
    parameters = deepcopy(SCHEMAS[index]["parameters"])
    expected = {"name": SCHEMAS[index]["name"], "description": SCHEMAS[index]["description"]}
    if provider == "openai":
        expected.update(type="function", parameters=parameters)
    else:
        parameters["additionalProperties"] = False
        expected["input_schema"] = parameters
    render = {"openai": responses_tool_schemas, "anthropic": messages_tool_schemas}[provider]
    assert render((definition,))[0] == expected


def test_anthropic_override_does_not_mutate_source_parameters():
    parameters = deepcopy(SEARCH.parameters)
    parameters["additionalProperties"] = True
    tool = replace(SEARCH, parameters=parameters)
    schema = messages_tool_schemas((tool,))[0]
    assert schema["input_schema"]["additionalProperties"] is False
    assert parameters["additionalProperties"] is True


@pytest.mark.parametrize(
    "render,key", [(responses_tool_schemas, "parameters"), (messages_tool_schemas, "input_schema")]
)
def test_nested_native_schema_mutation_is_isolated_from_later_turns(render, key):
    from parallel_search_evals.tools import ToolDispatcher
    from parallel_search_evals.tools.schemas import TOOLS

    tools = ToolDispatcher(TOOLS)
    expected = deepcopy(render(tools.available))
    exposed = render(tools.available)
    exposed[0][key]["properties"]["objective"]["type"] = "changed"
    exposed[1][key]["required"].append("changed")
    assert render(tools.available) == expected
    assert render(ToolDispatcher(TOOLS).available) == expected
