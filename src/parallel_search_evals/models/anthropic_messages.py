"""Anthropic Messages owns grouped tool results and native assistant blocks.

Thinking signatures and unknown block fields remain in replay dictionaries;
parsing extracts current text and tool controls without reconstructing blocks.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal, Self, cast

import httpx
import orjson
from anthropic import AsyncAnthropic, omit
from anthropic.types import MessageParam, ToolChoiceParam, ToolParam
from pydantic import BaseModel, ConfigDict, Field

from parallel_search_evals import config
from parallel_search_evals.models.base import ModelClient, ModelReply, ResponseBlock, ResponseText
from parallel_search_evals.tools import ToolDispatcher, ToolSpec
from parallel_search_evals.tools.schemas import (
    ToolInvocation,
    ToolReply,
)


class MessagesPayload(BaseModel):
    stop_reason: Literal["tool_use", "end_turn", "stop_sequence"]
    content: list[dict[str, Any]]


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class MessagesHistory(BaseModel):
    """Own native replay state and the latest parsed assistant text."""

    model_config = ConfigDict(extra="forbid")

    messages: list[AnthropicMessage] = []
    system: str = ""
    assistant_text: str = Field(strict=True)

    def append_question(self, content: str) -> Self:
        self.messages.append(AnthropicMessage(role="user", content=[{"type": "text", "text": content}]))
        return self

    def append_answer(self, content: str) -> Self:
        self.messages.append(AnthropicMessage(role="assistant", content=[{"type": "text", "text": content}]))
        self.assistant_text = content
        return self

    def accept_reply(self, output: ModelReply) -> Self:
        self.messages.append(AnthropicMessage(role="assistant", content=output.native))
        self.assistant_text = output.text
        return self

    def append_tool_outputs(self, results: list[ToolReply], continuation: str = "") -> Self:
        content: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.output,
                "is_error": result.is_error,
            }
            for result in results
        ]
        if continuation:
            content.append({"type": "text", "text": continuation})
        self.messages.append(AnthropicMessage(role="user", content=content))
        return self

    def request_messages(self) -> list[MessageParam]:
        return cast(list[MessageParam], self.model_dump(include={"messages"})["messages"])


def messages_tool_schemas(actions: Sequence[ToolSpec]) -> list[ToolParam]:
    """Copy trusted declarations into native Messages tools with closed inputs."""
    return cast(
        list[ToolParam],
        [
            {
                "name": action.name,
                "description": action.description,
                "input_schema": {**deepcopy(action.parameters), "additionalProperties": False},
            }
            for action in actions
        ],
    )


class MessagesClient(ModelClient):
    def __init__(
        self, model: str, *, client: AsyncAnthropic, max_tokens: int, effort: str | None, thinking_budget: int | None
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking_budget = thinking_budget
        self.model = model

    async def aclose(self) -> None:
        await self.client.close()

    def new_history(self) -> MessagesHistory:
        return MessagesHistory(assistant_text="")

    async def generate(self, history: MessagesHistory, tools: ToolDispatcher) -> ModelReply:
        # Anthropic thinking requests require automatic tool selection.
        native_choice = {
            "auto": "auto",
            "required": "auto" if self.thinking_budget else "any",
            "none": "none",
        }[tools.tool_choice]

        raw = await self.client.messages.with_raw_response.create(
            model=self.model,
            timeout=httpx.Timeout(config.CALL_SECONDS, connect=config.CONNECT_SECONDS),
            extra_body={"output_config": {"effort": self.effort} if self.effort is not None else {}},
            max_tokens=self.max_tokens,
            system=history.system,
            messages=history.request_messages(),
            tools=messages_tool_schemas(tools.available),
            tool_choice=cast(ToolChoiceParam, {"type": native_choice}),
            thinking={"type": "enabled", "budget_tokens": self.thinking_budget} if self.thinking_budget else omit,
        )
        return self._parse_response(orjson.loads(raw.http_response.content))

    def _parse_response(self, response: Any) -> ModelReply:
        parsed = MessagesPayload.model_validate(response, strict=True)
        texts: list[str] = []
        calls: list[ToolInvocation] = []

        for block in parsed.content:
            match ResponseBlock.model_validate(block, strict=True).type:
                case "tool_use":
                    calls.append(ToolInvocation.model_validate({**block, "type": "tool_call"}, strict=True))
                case "text":
                    texts.append(ResponseText.model_validate(block, strict=True).text)

        return ModelReply(texts=texts, tool_calls=calls, native=parsed.content)
