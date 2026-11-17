"""Google generation preserves native contents and parts across local tools.

Raw response bodies retain signed thought parts and unknown fields. Continuation
requests reuse the original body and exact token; local call IDs identify parts
across fragments without replacing the provider's native function-call IDs.
"""

import json
from copy import deepcopy
from typing import Annotated, Any, Literal, Self, cast

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from parallel_search_evals import config
from parallel_search_evals.models.base import ModelClient, ModelReply
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import ToolInvocation, ToolReply


class GeminiMessage(BaseModel):
    role: str
    parts: list[dict[str, Any]]


class GoogleHistory(BaseModel):
    """Own native replay state and the latest parsed assistant text."""

    model_config = ConfigDict(extra="forbid")

    messages: list[GeminiMessage] = []
    system: str = ""
    assistant_text: str = Field(strict=True)

    def append_question(self, content: str) -> Self:
        self.messages.append(GeminiMessage(role="user", parts=[{"text": content}]))
        return self

    def append_answer(self, content: str) -> Self:
        self.messages.append(GeminiMessage(role="model", parts=[{"text": content}]))
        self.assistant_text = content
        return self

    def accept_reply(self, output: ModelReply) -> Self:
        self.messages.append(GeminiMessage(role="model", parts=output.native))
        self.assistant_text = output.text
        return self

    def append_tool_outputs(self, results: list[ToolReply], continuation: str = "") -> Self:
        calls = {
            f"{len(self.messages) - 1}:{i}": p["functionCall"]
            for i, p in enumerate(self.messages[-1].parts)
            if "functionCall" in p
        }
        self.messages.append(
            GeminiMessage(
                role="user",
                parts=[
                    {
                        "functionResponse": {
                            "name": calls[r.tool_call_id]["name"],
                            "response": {"error" if r.is_error else "output": r.output},
                            **({"id": calls[r.tool_call_id]["id"]} if calls[r.tool_call_id].get("id") else {}),
                        }
                    }
                    for r in results
                ],
            )
        )
        # Keep final instructions in their own turn after grouped tool outputs.
        if continuation:
            self.append_question(continuation)
        return self

    def request_messages(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.model_dump(include={"messages"})["messages"])


type NonblankString = Annotated[str, Field(strict=True, min_length=1, pattern=r"[^\s\x1c-\x1f]")]


class GeminiFunction(BaseModel):
    id: Annotated[str, Field(strict=True, pattern=r"^$|[^\s\x1c-\x1f]")] | None = None
    name: NonblankString
    args: dict[str, Any] = {}


class GeminiPart(BaseModel):
    text: str | None = None
    thought: bool = False
    function_call: GeminiFunction | None = Field(None, alias="functionCall")
    thought_signature: str | None = Field(None, alias="thoughtSignature")


class GeminiContent(BaseModel):
    role: Literal["model"]
    parts: list[GeminiPart]


class GeminiCandidate(BaseModel):
    finish_reason: Literal["STOP", "CONTINUATION"] = Field(alias="finishReason")
    continuation_token: NonblankString | None = Field(None, alias="continuationToken")
    content: GeminiContent


class GeminiFeedback(BaseModel):
    block_reason: None = Field(None, alias="blockReason")


class GeminiResponse(BaseModel):
    candidates: Annotated[list[GeminiCandidate], Field(min_length=1, max_length=1)]
    prompt_feedback: GeminiFeedback = Field(default=GeminiFeedback(blockReason=None), alias="promptFeedback")


async def _set_connect_timeout(request: httpx.Request) -> None:
    # Set the connect deadline after the SDK applies its per-request timeout.
    request.extensions["timeout"]["connect"] = float(config.CONNECT_SECONDS)


def _native_http_options(base_url: str, api_version: str) -> types.HttpOptions:
    return types.HttpOptions(
        base_url=base_url,
        api_version=api_version,
        retry_options=types.HttpRetryOptions(attempts=config.GOOGLE_ATTEMPTS),
        client_args={"follow_redirects": False},
        async_client_args={
            "follow_redirects": False,
            "transport": httpx.AsyncHTTPTransport(),
            "event_hooks": {"request": [_set_connect_timeout]},
        },
    )


class GoogleClient(ModelClient):
    def __init__(
        self,
        model: str,
        *,
        client: genai.Client,
        owned: bool = False,
        max_tokens: int,
        thinking_level: str | None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = thinking_level
        self.owned = owned
        self.client = client

    @classmethod
    def from_settings(cls, model_settings: ModelSettings) -> Self:
        if model_settings.provider == "gemini":
            client = genai.Client(
                vertexai=False,
                api_key=model_settings.load_api_key(),
                http_options=_native_http_options("https://generativelanguage.googleapis.com", "v1beta"),
            )
        else:
            project, location = model_settings.vertex_project, model_settings.vertex_location
            if not project or not location:
                raise ValueError("Vertex requires explicit project and location")
            host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=_native_http_options(f"https://{host}", "v1beta1"),
            )
        return cls(
            model_settings.model,
            client=client,
            owned=True,
            max_tokens=model_settings.max_tokens,
            thinking_level=model_settings.effort,
        )

    async def aclose(self) -> None:
        if self.owned:
            try:
                await self.client.aio.aclose()
            finally:
                self.client.close()

    def new_history(self) -> GoogleHistory:
        return GoogleHistory(assistant_text="")

    async def generate(self, history: GoogleHistory, tools: ToolDispatcher) -> ModelReply:
        definitions = [
            {"name": tool.name, "description": tool.description, "parameters": deepcopy(tool.parameters)}
            for tool in tools.available
        ]
        generation: dict[str, Any] = {"maxOutputTokens": self.max_tokens, "candidateCount": 1}
        if self.effort is not None:
            generation["thinkingConfig"] = {"thinkingLevel": self.effort}
        body = {
            "contents": history.request_messages(),
            "generationConfig": generation,
            "systemInstruction": {"parts": [{"text": history.system}]},
            "tools": [{"functionDeclarations": definitions}] if definitions else [],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": {"auto": "AUTO", "none": "NONE", "required": "ANY"}[tools.tool_choice]
                }
            },
        }
        return await self._generate_content(body, str(len(history.messages)))

    async def _request(self, body: dict[str, Any]) -> Any:
        # Keep native signatures in extra_body and execute tools outside the SDK.
        # Handle continuation here so opaque fields survive typed SDK processing.
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents="native contents supplied in extra_body",
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(extra_body=body, timeout=config.CALL_SECONDS * 1000),
                should_return_http_response=True,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                automatic_continuation=False,
            ),
        )
        raw_body = getattr(response.sdk_http_response, "body", None)
        if raw_body is None:
            raise NativeResponseError("Gemini SDK returned no native response body")
        return json.loads(raw_body)

    async def _generate_content(self, body: dict[str, Any], call_prefix: str) -> ModelReply:
        combined = ModelReply(texts=[], tool_calls=[], native=[])
        seen_tokens: set[str] = set()
        seen_call_ids: set[str] = set()
        for _ in range(config.MAX_CONTINUATION_HOPS):
            native = await self._request(body)
            output = self._parse_response(native, seen_call_ids, call_prefix, len(combined.native))
            combined.native.extend(output.native)
            combined.texts.extend(output.texts)
            combined.tool_calls.extend(output.tool_calls)
            candidate = native["candidates"][0]
            if candidate["finishReason"] == "STOP":
                if seen_tokens:
                    # Join continuation fragments directly because a token can divide a word.
                    combined.texts = ["".join(combined.texts)]
                return combined
            token = candidate.get("continuationToken")
            if token is None or token in seen_tokens:
                raise NativeResponseError("Missing or repeated Gemini continuation token")
            seen_tokens.add(token)
            # Reuse the original request body with the exact continuation token.
            body["continuationToken"] = token
        raise NativeResponseError("Gemini continuation hop limit exhausted")

    def _parse_response(
        self, response: dict[str, Any], seen_call_ids: set[str], call_prefix: str = "", part_offset: int = 0
    ) -> ModelReply:
        try:
            parsed = GeminiResponse.model_validate(response, strict=True)
        except ValidationError:
            raise NativeResponseError("Malformed or incomplete Gemini response") from None
        parts = response["candidates"][0]["content"]["parts"]
        texts: list[str] = []
        calls: list[ToolInvocation] = []
        for index, part in enumerate(parsed.candidates[0].content.parts):
            call = part.function_call
            if call is not None:
                calls.append(ToolInvocation(id=f"{call_prefix}:{part_offset + index}", name=call.name, input=call.args))
                if call.id:
                    if call.id in seen_call_ids:
                        raise NativeResponseError("Gemini returned duplicate function call IDs")
                    seen_call_ids.add(call.id)
            elif part.text is not None and not part.thought:
                texts.append(part.text)
        return ModelReply(texts=texts, tool_calls=calls, native=parts)
