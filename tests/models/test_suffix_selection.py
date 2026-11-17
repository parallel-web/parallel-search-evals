"""Suffix resolution is delimiter-sensitive, route-specific and applied once."""

import pytest
from pydantic import ValidationError

from parallel_search_evals.models.settings import ModelSettings


@pytest.mark.parametrize(
    "model,provider,effort,physical,resolved_effort,thinking",
    [
        ("low", "openai", "default", "low", "high", None),
        ("gpt-low-low", "openai", "default", "gpt-low", "low", None),
        ("gpt-low-", "openai", "default", "gpt-low-", "high", None),
        ("gpt-LOW", "openai", "default", "gpt-LOW", "high", None),
        ("gpt-low", "openai", "high", "gpt", "high", None),
        ("gpt-low", "openai", "none", "gpt", None, None),
        ("gpt-low", "openrouter", "default", "gpt-low", "high", None),
        ("gpt-low", "anthropic", "default", "gpt-low", "max", None),
        ("claude-thinking-thinking", None, "default", "claude-thinking", "max", 1024),
        ("thinking", "anthropic", "default", "thinking", "max", None),
    ],
)
def test_native_suffix_resolution_boundaries(model, provider, effort, physical, resolved_effort, thinking):
    model_settings = ModelSettings.from_selection({"model": model, "provider": provider, "effort": effort})
    assert model_settings.model == physical
    assert model_settings.effort == resolved_effort
    assert model_settings.thinking_budget == thinking


@pytest.mark.parametrize("provider,model", [("openai", "-low"), ("anthropic", "-thinking")])
def test_alias_without_physical_model_remains_invalid(provider, model):
    with pytest.raises(ValidationError):
        ModelSettings.from_selection({"provider": provider, "model": model})


def test_applied_dump_keeps_alias_looking_physical_model_without_resolving_again():
    model_settings = ModelSettings.from_selection({"model": "gpt-low-low", "effort": "none"})
    restored = ModelSettings.model_validate(model_settings.model_dump())
    assert restored.model == "gpt-low" and restored.effort is None
