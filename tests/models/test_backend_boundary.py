"""OpenRouter owns selected backend constraints; unrelated routes ignore them."""

import json

import pytest

import parallel_search_evals.models as providers
from parallel_search_evals.models.settings import ModelSettings


@pytest.mark.parametrize(
    "backends", [("",), (" \t\n",), ("\x1c",), ("\x1d",), ("\x1e",), ("\x1f",), ("exact", "exact")]
)
def test_selected_backend_constraints_reject_before_any_sdk(monkeypatch, backends):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")

    def forbidden(**kwargs):
        raise AssertionError("Invalid backend selection reached native SDK construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden)
    model_settings = ModelSettings.from_selection(
        {"model": "vendor/opaque-high", "provider": "openrouter", "openrouter_backends": backends}
    )
    before = model_settings.model_dump()
    with pytest.raises(ValueError):
        providers.open_model(model_settings)
    assert model_settings.model_dump() == before


@pytest.mark.parametrize("backends", [(), ("",), (" \t\x1c",), ("same", "same")])
@pytest.mark.parametrize("provider", ["openai", "anthropic", "unbiased", "compatible", "gemini", "vertex"])
def test_irrelevant_backend_options_do_not_tighten_other_route_admission(monkeypatch, backends, provider):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    model_settings = ModelSettings.from_selection(
        {
            "model": "native",
            "provider": provider,
            "max_tokens": 64,
            "base_url": "https://custom.example/v1",
            "openrouter_backends": backends,
        }
    )
    model_settings.validate_selected_route()
    assert model_settings.openrouter_backends == backends


@pytest.mark.parametrize("backends", [(" exact ",), ("Case", "case"), ("tag", " tag "), (" \x1cbackend\x1f ",)])
def test_selected_backend_values_remain_exact_ordered_and_unmodified(monkeypatch, backends):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    selection = {"model": "vendor/opaque-high", "provider": "openrouter", "openrouter_backends": backends}
    original = dict(selection)
    model_settings = ModelSettings.from_selection(selection)
    before = json.dumps(model_settings.model_dump())
    model_settings.validate_selected_route()
    assert model_settings.openrouter_backends == backends and selection == original
    assert json.dumps(model_settings.model_dump()) == before


@pytest.mark.parametrize("backends", [(), []])
def test_empty_openrouter_backend_selection_delegates_routing(backends):
    model_settings = ModelSettings.from_selection(
        {"model": "openrouter/auto", "provider": "openrouter", "openrouter_backends": backends}
    )
    model_settings.validate_selected_route()
    assert model_settings.openrouter_backends == ()
