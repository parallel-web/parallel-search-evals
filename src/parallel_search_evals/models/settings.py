"""Validate model selections separately from restoring saved resolved settings.

Selection applies route inference, recognized model aliases and provider defaults;
model_validate restores settings without repeating those transformations.
Model/backend output ceilings apply only after field validation and only when no
explicit cap was supplied. Credentials are read only for the selected route.
Compatible endpoints reject embedded credentials and allow plaintext solely on
localhost or literal loopback addresses; native tenant headers stay route-owned.
"""

import os
from ipaddress import ip_address
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from parallel_search_evals import config
from parallel_search_evals.config import DEFAULTS, KEYS, MODEL_BACKEND_MAX_TOKENS, MODEL_MAX_TOKENS
from parallel_search_evals.config import ENDPOINTS as ENDPOINTS

type OpenRouterBackends = tuple[Annotated[str, Field(strict=True, min_length=1, pattern=r"[^\s\x1c-\x1f]")], ...]
OPENROUTER_BACKENDS: TypeAdapter[OpenRouterBackends] = TypeAdapter(OpenRouterBackends)


def _resolve_selection_defaults(values: dict[str, Any], model: str, provider: str) -> None:
    default_cap, default_effort = DEFAULTS.get(provider, (None, None))
    if values.get("max_tokens") is None:
        values["max_tokens"] = default_cap
    effort = values.get("effort", "default")
    if provider == "openai" and model.endswith(("-minimal", "-low", "-medium", "-high", "-xhigh", "-max")):
        model, _, suffix = model.rpartition("-")
        if effort == "default":
            effort = suffix
    if provider == "anthropic" and model.endswith("-thinking"):
        model = model.removesuffix("-thinking")
        values["thinking_budget"] = config.ANTHROPIC_THINKING_BUDGET
    values["model"] = model
    if effort in ("default", "none"):
        effort = {"default": default_effort, "none": None}[effort]
    values["effort"] = effort


def _validate_compatible_url(value: str | None) -> None:
    parsed = urlsplit(value)
    _ = parsed.port
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any((parsed.username is not None, parsed.password is not None, parsed.query, parsed.fragment))
    ):
        raise ValueError("Compatible base_url requires HTTP(S), a host, and no credentials/query/fragment")
    # Restrict plaintext endpoints to localhost or literal loopback addresses.
    if parsed.scheme == "http" and parsed.hostname != "localhost":
        try:
            loopback = ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("Remote compatible base_url requires HTTPS")


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    model: str = Field(strict=True, min_length=1, pattern=r"[^\s\x1c-\x1f]")
    provider: Literal["openai", "anthropic", "openrouter", "unbiased", "compatible", "gemini", "vertex"]
    max_tokens: int = Field(strict=True, gt=0)
    openrouter_backends: tuple[str, ...] = ()
    base_url: str | None = None
    api: str = "responses"
    api_key_env: str | None = Field(None, strict=True, pattern=r"^[A-Z_][A-Z0-9_]*$")
    vertex_project: str | None = None
    vertex_location: str | None = None

    effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None
    thinking_budget: Literal[1024] | None = None

    @classmethod
    def from_selection(cls, data: Any) -> Self:
        """Apply selection defaults; model_validate restores already resolved settings."""
        if not isinstance(data, dict):
            return cls.model_validate(data)
        values = dict(data)
        model = values.get("model")
        if not isinstance(model, str):
            return cls.model_validate(values)
        provider = values.get("provider")
        for prefix, route in (("gpt", "openai"), ("claude", "anthropic")):
            if provider is None and model.startswith(prefix):
                provider = route
        values["provider"] = provider
        if not isinstance(provider, str):
            return cls.model_validate(values)
        cap = values.get("max_tokens")
        _resolve_selection_defaults(values, model, provider)
        model_settings = cls.model_validate(values)
        if cap is None:
            maximum = MODEL_BACKEND_MAX_TOKENS.get(
                (model_settings.provider, model_settings.model, model_settings.openrouter_backends),
                MODEL_MAX_TOKENS.get((model_settings.provider, model_settings.model), model_settings.max_tokens),
            )
            return model_settings.model_copy(update={"max_tokens": maximum})
        return model_settings

    def load_api_key(self) -> str:
        name = (self.api_key_env if self.provider == "compatible" else None) or KEYS[self.provider]
        value = os.environ.get(name, "")
        if not value.strip():
            raise ValueError(f"{name} is required")
        return value

    def validate_selected_route(self) -> None:
        if self.provider == "openrouter":
            OPENROUTER_BACKENDS.validate_python(self.openrouter_backends)
            if len(set(self.openrouter_backends)) != len(self.openrouter_backends):
                raise ValueError("OpenRouter requires unique backend tags")
        if self.provider == "compatible":
            if self.api not in {"responses", "chat"}:
                raise ValueError("Compatible api must be responses or chat")
            _validate_compatible_url(self.base_url)
        if self.provider in {"gemini", "vertex"} and self.effort in {"xhigh", "max"}:
            raise ValueError("Google effort must be minimal, low, medium or high")
