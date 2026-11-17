"""Shared parsed replies, persisted OpenAI replay envelopes, and SDK ownership.

Native replay items stay opaque so signed reasoning and nullable future fields
survive persistence. The OpenAI transport owns SDK configuration and closure;
concrete clients own their protocol requests and initialize their own histories.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, Self, cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputItemParam
from pydantic import BaseModel, ConfigDict, Field

from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import ToolInvocation, ToolReply


class ResponseBlock(BaseModel):
    type: str


class ResponseText(ResponseBlock):
    text: str


class FunctionArguments(BaseModel):
    name: str
    arguments: str


class ModelReply(BaseModel):
    """Parsed answer and tool controls accompany complete native response items."""

    texts: list[str]
    tool_calls: list[ToolInvocation]
    native: list[dict[str, Any]]

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


class ResponsesHistoryEntry(BaseModel):
    native: dict[str, Any]


class ResponsesHistory(BaseModel):
    """Own native replay state and the latest parsed assistant text."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ResponsesHistoryEntry] = []
    system: str = ""
    assistant_text: str = Field(strict=True)
    api: Literal["responses", "chat"] = "responses"

    def append_question(self, content: str) -> Self:
        self.messages.append(ResponsesHistoryEntry(native={"role": "user", "content": content}))
        return self

    def append_answer(self, content: str) -> Self:
        self.messages.append(ResponsesHistoryEntry(native={"role": "assistant", "content": content}))
        self.assistant_text = content
        return self

    def accept_reply(self, output: ModelReply) -> Self:
        self.messages.extend(ResponsesHistoryEntry(native=item) for item in output.native)
        self.assistant_text = output.text
        return self

    def append_tool_outputs(self, results: list[ToolReply], continuation: str = "") -> Self:
        self.messages.extend(
            ResponsesHistoryEntry(
                native={"role": "tool", "tool_call_id": item.tool_call_id, "content": item.output}
                if self.api == "chat"
                else {"type": "function_call_output", "call_id": item.tool_call_id, "output": item.output}
            )
            for item in results
        )
        if continuation:
            self.append_question(continuation)
        return self

    def request_messages(self) -> list[ResponseInputItemParam]:
        items = [message.native for message in self.messages]
        if self.api == "chat" and self.system:
            items.insert(0, {"role": "system", "content": self.system})
        return cast(list[ResponseInputItemParam], items)


class ModelClient(ABC):
    model: str
    max_tokens: int
    effort: str | None

    @abstractmethod
    async def aclose(self) -> None: ...

    @abstractmethod
    def new_history(self) -> Any:
        """Initialize provider history with an empty answer."""
        ...

    @abstractmethod
    async def generate(self, history: Any, tools: ToolDispatcher) -> ModelReply: ...


class OpenAITransport(ModelClient):
    """Own the SDK connection shared by independent OpenAI-compatible protocols."""

    def __init__(
        self,
        model: str,
        *,
        client: AsyncOpenAI,
        max_tokens: int,
        effort: str | None,
        openrouter_backends: tuple[str, ...] = (),
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.openrouter_backends = openrouter_backends
        self.effort = effort
        self.model = model

    async def aclose(self) -> None:
        await self.client.close()
