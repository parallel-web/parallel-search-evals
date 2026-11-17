import asyncio
import json
from collections.abc import Sequence
from typing import cast

import httpx
import orjson
from parallel import APIError
from pydantic import ValidationError

from parallel_search_evals.config import MAX_TOOL_CALLS
from parallel_search_evals.runtime import NativeResponseError, join_tasks
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import ToolChoice, ToolInvocation, ToolReply, ToolSpec


class ToolDispatcher:
    def __init__(
        self,
        actions: Sequence[ToolSpec],
        tool_choice: ToolChoice = ToolChoice.AUTO,
        max_calls: int = MAX_TOOL_CALLS,
        *,
        client: ParallelClient | None = None,
    ) -> None:
        self.actions = {action.name: action for action in actions}
        self.client = client
        self.tool_choice = tool_choice
        self.remaining_calls = max_calls

    @property
    def available(self) -> tuple[ToolSpec, ...]:
        if self.tool_choice == ToolChoice.NONE:
            return ()
        return tuple(self.actions.values())

    async def _dispatch_one(self, call: ToolInvocation) -> ToolReply:
        """Reject invalid arguments without spending budget.

        Reserve a valid call before awaiting so concurrent calls cannot oversubscribe.
        """
        try:
            action = self.actions[call.name]
        except KeyError as error:
            return ToolReply(tool_call_id=call.id, output=f"Error: {error}", is_error=True)
        try:
            arguments = action.input_schema.model_validate(call.input)
        except ValidationError as error:
            return ToolReply(tool_call_id=call.id, output=f"Error: {error}", is_error=True)
        if self.remaining_calls <= 0:
            return ToolReply(tool_call_id=call.id, output="Tool budget exhausted", is_error=True)
        self.remaining_calls -= 1
        try:
            output = orjson.dumps(
                await cast(ParallelClient, self.client).request(
                    action.endpoint, arguments.model_dump(exclude_none=True)
                )
            ).decode()
        except (
            APIError,
            httpx.HTTPError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            NativeResponseError,
            ValidationError,
        ) as error:
            return ToolReply(tool_call_id=call.id, output=f"Error: {error}", is_error=True)
        return ToolReply(tool_call_id=call.id, output=output)

    async def execute(self, calls: list[ToolInvocation]) -> list[ToolReply]:
        tasks = [asyncio.create_task(self._dispatch_one(call)) for call in calls]
        return await join_tasks(tasks)
