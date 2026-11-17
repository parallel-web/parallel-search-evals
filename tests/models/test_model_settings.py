"""Invalid combinations stop before any native client can be created."""

import pytest

import parallel_search_evals.models as providers
from parallel_search_evals.models.settings import ModelSettings


@pytest.mark.parametrize(
    "options",
    [
        {"provider": "unknown"},
        {"provider": "openai", "max_tokens": 0},
        {"provider": "compatible", "base_url": "https://host", "api": "invalid"},
        {"provider": "gemini", "max_tokens": -1},
        {"provider": "vertex", "vertex_project": "project"},
        {"provider": "compatible", "base_url": "https://user:password@host"},
    ],
)
def test_invalid_selection_cannot_create_a_client(monkeypatch, options):
    def unexpected(**kwargs):
        raise AssertionError("Invalid settings reached native client construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", unexpected)
    monkeypatch.setattr(providers, "AsyncAnthropic", unexpected)
    with pytest.raises(ValueError):
        providers.open_model(ModelSettings.from_selection({"model": "exact-model", **options}))


@pytest.mark.parametrize("provider", ["unbiased", "compatible"])
def test_configuration_only_routes_require_explicit_cap(monkeypatch, provider):
    def forbidden(**kwargs):
        raise AssertionError("Missing required cap reached SDK construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden)
    monkeypatch.setenv("UNBIASED_API_KEY", "fixture")
    monkeypatch.setenv("COMPATIBLE_API_KEY", "fixture")
    with pytest.raises(ValueError, match="max_tokens"):
        providers.open_model(
            ModelSettings.from_selection({"model": "exact-model", "provider": provider, "base_url": "https://host"})
        )


@pytest.mark.parametrize("effort", ["unknown", "HIGH"])
def test_effort_is_not_silently_remapped(monkeypatch, effort):
    with pytest.raises(ValueError, match="effort"):
        providers.open_model(ModelSettings.from_selection({"model": "gpt-fixture", "effort": effort}))


@pytest.mark.parametrize("effort", [[], {}, ["default"], {"effort": "none"}])
def test_unhashable_effort_reaches_field_validation_without_mutating_selection(effort):
    selection = {"model": "gpt-fixture-high", "effort": effort}
    original = dict(selection)
    with pytest.raises(ValueError, match="literal_error"):
        ModelSettings.from_selection(selection)
    assert selection == original
