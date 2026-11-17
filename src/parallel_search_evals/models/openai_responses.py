"""Responses requests retain opaque output items, including encrypted reasoning.

The parser admits completed native envelopes and extracts only current answer
text and tool controls; replay retains every original output item unchanged.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal, cast

import httpx
import orjson
from openai import NOT_GIVEN
from openai.types.responses import FunctionToolParam
from pydantic import BaseModel

from parallel_search_evals import config
from parallel_search_evals.models.base import (
    FunctionArguments,
    ModelReply,
    OpenAITransport,
    ResponseBlock,
    ResponsesHistory,
    ResponseText,
)
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher, ToolSpec
from parallel_search_evals.tools.schemas import ToolInvocation


class ResponsesMessage(BaseModel):
    content: list[dict[str, Any]]

    def answer_texts(self) -> list[str]:
        """Admit content in wire order, stopping immediately on refusal."""
        texts: list[str] = []
        for content in self.content:
            block = ResponseBlock.model_validate(content, strict=True)
            if block.type == "refusal":
                raise NativeResponseError("OpenAI refused the request")
            if block.type == "output_text":
                text = ResponseText.model_validate(content, strict=True).text
                if text:
                    texts.append(text)
        return texts


class ResponsesFunctionCall(FunctionArguments):
    call_id: str

    def invocation(self) -> ToolInvocation:
        """Decode native arguments through the shared invocation admission type."""
        return ToolInvocation(id=self.call_id, name=self.name, input=orjson.loads(self.arguments))


class ResponsesPayload(BaseModel):
    status: Literal["completed"]
    output: list[dict[str, Any]]

    def reply(self) -> ModelReply:
        """Project current controls while retaining every opaque replay item."""
        texts: list[str] = []
        calls: list[ToolInvocation] = []
        for item in self.output:
            block = ResponseBlock.model_validate(item, strict=True)
            if block.type == "function_call":
                calls.append(ResponsesFunctionCall.model_validate(item, strict=True).invocation())
            elif block.type == "message":
                texts.extend(ResponsesMessage.model_validate(item, strict=True).answer_texts())
        return ModelReply(texts=texts, tool_calls=calls, native=self.output)


def responses_tool_schemas(actions: Sequence[ToolSpec]) -> list[FunctionToolParam]:
    """Copy trusted declarations into native Responses function tools."""
    return cast(
        list[FunctionToolParam],
        [
            {
                "name": action.name,
                "description": action.description,
                "type": "function",
                "parameters": deepcopy(action.parameters),
            }
            for action in actions
        ],
    )


class ResponsesClient(OpenAITransport):
    def new_history(self) -> ResponsesHistory:
        return ResponsesHistory(assistant_text="")

    async def generate(self, history: ResponsesHistory, tools: ToolDispatcher) -> ModelReply:
        raw = await self.client.responses.with_raw_response.create(  # type: ignore[call-overload]
            model=self.model,
            timeout=httpx.Timeout(config.CALL_SECONDS, connect=config.CONNECT_SECONDS),
            input=history.request_messages(),
            instructions=history.system or NOT_GIVEN,
            tools=responses_tool_schemas(tools.available),
            tool_choice=NOT_GIVEN if tools.tool_choice == "auto" else tools.tool_choice,
            reasoning={"effort": self.effort} if self.effort is not None else NOT_GIVEN,
            extra_body={
                "provider": {
                    "only": list(self.openrouter_backends),
                    "order": list(self.openrouter_backends),
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            }
            if self.openrouter_backends
            else None,
            store=False,
            max_output_tokens=self.max_tokens,
            **(
                {"prompt_cache_key": config.PROMPT_CACHE_KEY, "include": ["reasoning.encrypted_content"]}
                if self.client.base_url.host in {"api.openai.com", "openrouter.ai"}
                else {}
            ),
        )
        return parse_responses_reply(orjson.loads(raw.http_response.content))


def parse_responses_reply(response: Any) -> ModelReply:
    return ResponsesPayload.model_validate(response, strict=True).reply()
