"""Build one selected model client with isolated credentials and SDK ownership.

Route validation precedes credential loading and client construction. Compatible
endpoints omit OpenAI account headers; SDK retries and redirects are disabled
so request attempts and credential destinations remain under evaluator control.
"""

from anthropic import AsyncAnthropic
from anthropic import DefaultAsyncHttpxClient as AnthropicHttpClient
from anthropic import omit as anthropic_omit
from openai import AsyncOpenAI, omit
from openai import DefaultAsyncHttpxClient as OpenAIHttpClient

from parallel_search_evals import config
from parallel_search_evals.models.anthropic_messages import MessagesClient, MessagesHistory
from parallel_search_evals.models.base import ModelClient, ModelReply, ResponsesHistory
from parallel_search_evals.models.chat_completions import ChatClient
from parallel_search_evals.models.gemini import GoogleClient, GoogleHistory
from parallel_search_evals.models.openai_responses import ResponsesClient
from parallel_search_evals.models.settings import ENDPOINTS, ModelSettings

ModelHistory = ResponsesHistory | MessagesHistory | GoogleHistory


def open_model(model_settings: ModelSettings) -> ModelClient:
    model_settings.validate_selected_route()
    if model_settings.provider in {"gemini", "vertex"}:
        return GoogleClient.from_settings(model_settings)
    if model_settings.provider == "anthropic":
        client = AsyncAnthropic(
            api_key=model_settings.load_api_key(),
            base_url=ENDPOINTS[model_settings.provider],
            # The omission sentinel suppresses the SDK default Authorization header.
            default_headers={"Authorization": anthropic_omit},  # type: ignore[dict-item]
            max_retries=config.SDK_MAX_RETRIES,
            http_client=AnthropicHttpClient(follow_redirects=False),
        )
        return MessagesClient(
            model_settings.model,
            client=client,
            max_tokens=model_settings.max_tokens,
            effort=model_settings.effort,
            thinking_budget=model_settings.thinking_budget,
        )
    response_client = AsyncOpenAI(
        api_key=model_settings.load_api_key(),
        base_url=model_settings.base_url
        if model_settings.provider == "compatible"
        else ENDPOINTS[model_settings.provider],
        default_headers={"OpenAI-Organization": omit, "OpenAI-Project": omit}  # type: ignore[dict-item]
        if model_settings.provider != "openai"
        else None,
        max_retries=config.SDK_MAX_RETRIES,
        http_client=OpenAIHttpClient(follow_redirects=False),
    )
    model_type = (
        ChatClient if model_settings.provider == "compatible" and model_settings.api == "chat" else ResponsesClient
    )
    return model_type(
        model_settings.model,
        client=response_client,
        max_tokens=model_settings.max_tokens,
        effort=model_settings.effort,
        openrouter_backends=model_settings.openrouter_backends if model_settings.provider == "openrouter" else (),
    )


__all__ = [
    "MessagesHistory",
    "MessagesClient",
    "ModelClient",
    "ModelHistory",
    "ModelReply",
    "ResponsesHistory",
    "ResponsesClient",
    "ModelSettings",
    "open_model",
]
