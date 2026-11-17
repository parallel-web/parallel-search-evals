"""Adapt native Chat Completions messages to the shared answer loop.

Chat tools use the nested function declaration shape and role=tool replies;
they are not Responses API items. Preserve the original assistant message,
including unknown fields, for later replay and persistence. Parsed text and
calls are separate controls: a refusal, incomplete finish status or mismatch
between stop status and tool calls must fail instead of producing a grade.
The shared OpenAI transport owns only SDK settings and client closure.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Annotated, Any, Literal, cast

import httpx
import orjson
from openai import omit
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from pydantic import BaseModel, Field

from parallel_search_evals import config
from parallel_search_evals.models.base import FunctionArguments, ModelReply, OpenAITransport, ResponsesHistory
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher, ToolSpec
from parallel_search_evals.tools.schemas import ToolInvocation


class ChatFunctionCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionArguments


class ChatMessage(BaseModel):
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatFunctionCall] | None = None


class ChatChoice(BaseModel):
    finish_reason: str
    message: ChatMessage


class ChatPayload(BaseModel):
    choices: Annotated[list[ChatChoice], Field(min_length=1, max_length=1)]


def chat_tool_schemas(actions: Sequence[ToolSpec]) -> list[ChatCompletionFunctionToolParam]:
    """Copy trusted declarations into native Chat Completions function tools."""
    return cast(
        list[ChatCompletionFunctionToolParam],
        [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description,
                    "parameters": deepcopy(action.parameters),
                },
            }
            for action in actions
        ],
    )


class ChatClient(OpenAITransport):
    def new_history(self) -> ResponsesHistory:
        return ResponsesHistory(assistant_text="", api="chat")

    async def generate(self, history: ResponsesHistory, tools: ToolDispatcher) -> ModelReply:
        chat_tools = chat_tool_schemas(tools.available)
        raw = await self.client.chat.completions.with_raw_response.create(
            model=self.model,
            timeout=httpx.Timeout(config.CALL_SECONDS, connect=config.CONNECT_SECONDS),
            messages=cast(list[ChatCompletionMessageParam], history.request_messages()),
            tools=chat_tools if chat_tools else omit,
            tool_choice=tools.tool_choice.value,
            max_tokens=self.max_tokens,
            extra_body={"reasoning_effort": self.effort} if self.effort is not None else None,
            stream=False,
        )
        return parse_chat_reply(orjson.loads(raw.http_response.content))


def parse_chat_reply(native: Any) -> ModelReply:
    response = ChatPayload.model_validate(native, strict=True)
    choice = response.choices[0]
    if choice.finish_reason not in {"stop", "tool_calls"}:
        raise NativeResponseError("Chat response did not complete")
    message = choice.message
    if message.refusal:
        raise NativeResponseError("Chat provider refused the request")
    texts = [message.content] if message.content is not None else []
    calls: list[ToolInvocation] = []
    for call in message.tool_calls or []:
        calls.append(ToolInvocation(id=call.id, name=call.function.name, input=orjson.loads(call.function.arguments)))
    if (choice.finish_reason == "tool_calls") != bool(message.tool_calls):
        raise NativeResponseError("Chat stop status does not match native tool calls")
    return ModelReply(texts=texts, tool_calls=calls, native=[native["choices"][0]["message"]])
