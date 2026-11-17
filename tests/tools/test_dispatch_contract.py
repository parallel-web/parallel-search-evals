"""Dispatcher preserves input-validation error text and call budgets."""

import pytest
from pydantic import ValidationError

from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import EXTRACT, SEARCH, ToolInvocation


@pytest.mark.parametrize("tool", [SEARCH, EXTRACT])
@pytest.mark.parametrize("arguments", [{}, {"objective": "Evidence", "search_queries": []}])
async def test_validation_error_text_keeps_model_title_and_does_not_reserve(tool, arguments):
    with pytest.raises(ValidationError) as expected:
        tool.input_schema.model_validate(arguments)
    tools = ToolDispatcher([tool], max_calls=0)
    result = (await tools.execute([ToolInvocation(id="invalid", name=tool.name, input=arguments)]))[0]
    assert result.output == f"Error: {expected.value}"
    assert result.is_error and result.tool_call_id == "invalid"
    assert tools.remaining_calls == 0
