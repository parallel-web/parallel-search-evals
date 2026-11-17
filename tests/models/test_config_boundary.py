"""Selected numeric settings must fail before any native dispatch resource exists."""

import pytest

import parallel_search_evals.models as providers
from parallel_search_evals.models.settings import ModelSettings


def test_boolean_token_cap_is_not_a_positive_integer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-secret")

    def forbidden_client(**kwargs):
        pytest.fail("Boolean token cap reached SDK client construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden_client)
    with pytest.raises(ValueError):
        providers.open_model(
            ModelSettings.from_selection({"model": "selected", "provider": "openai", "max_tokens": True})
        )


@pytest.mark.parametrize("model", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_control_only_model_is_rejected_before_resources(monkeypatch, model):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-secret")

    def forbidden_client(**kwargs):
        pytest.fail("Blank model reached SDK client construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden_client)
    with pytest.raises(ValueError):
        providers.open_model(ModelSettings.from_selection({"model": model, "provider": "openai"}))


@pytest.mark.parametrize(
    ("provider", "model"),
    [("openai", "-high"), ("openai", " -high"), ("anthropic", "-thinking"), ("anthropic", " -thinking")],
)
def test_physical_model_must_remain_meaningful_after_suffix(monkeypatch, provider, model):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture")

    def unexpected(**kwargs):
        raise AssertionError("Blank physical model reached SDK construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", unexpected)
    monkeypatch.setattr(providers, "AsyncAnthropic", unexpected)
    with pytest.raises(ValueError):
        providers.open_model(ModelSettings.from_selection({"model": model, "provider": provider}))


@pytest.mark.parametrize(
    "selection",
    [
        {"model": "gpt-native-medium"},
        {"model": "gpt-native-medium-high"},
        {"model": "gpt-native-medium-high", "effort": "low"},
        {"model": "claude-native-thinking-thinking"},
        {"model": "gpt-native-medium", "effort": "low"},
        {"model": "gpt-native-high", "effort": "none"},
        {"model": "claude-native-thinking"},
        {"model": "claude-native-thinking", "effort": "none"},
        {"model": "vendor/native-high", "provider": "openrouter", "openrouter_backends": ["backend"]},
        {"model": "gemini-native", "provider": "gemini"},
        {"model": "native", "provider": "compatible", "max_tokens": 64, "base_url": "https://host"},
        {"model": "native", "provider": "unbiased", "max_tokens": 64},
    ],
)
def test_resolved_settings_are_authoritative_and_idempotent(selection):
    original = dict(selection)
    model_settings = ModelSettings.from_selection(selection)
    assert selection == original
    assert model_settings.provider is not None
    assert type(model_settings.max_tokens) is int and model_settings.max_tokens > 0
    assert model_settings.effort != "default"
    assert ModelSettings.model_validate(model_settings) is model_settings
    assert ModelSettings.model_validate(model_settings.model_dump()) == model_settings
    if selection["model"] == "gpt-native-medium-high":
        assert model_settings.model == "gpt-native-medium"
        assert model_settings.effort == selection.get("effort", "high")
    if selection["model"].endswith("-thinking"):
        assert model_settings.model == selection["model"].removesuffix("-thinking")
        assert model_settings.thinking_budget == 1024


@pytest.mark.parametrize("value", [None, [], "model", 42])
def test_bad_settings_root_uses_pydantic_validation(value):
    with pytest.raises(ValueError):
        ModelSettings.from_selection(value)
    with pytest.raises(ValueError):
        ModelSettings.model_validate(value)


def test_resolution_preserves_unknown_fields_for_rejection():
    with pytest.raises(ValueError, match="extra_forbidden"):
        ModelSettings.from_selection({"model": "gpt-native", "unexpected": True})


@pytest.mark.parametrize("provider", [[], {}, ["openai"], {"route": "openai"}, "unknown", False, 42])
def test_invalid_provider_selection_is_a_value_error_before_any_client(monkeypatch, provider):
    def forbidden(**kwargs):
        raise AssertionError("Invalid provider reached SDK construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(providers, "AsyncAnthropic", forbidden)
    selection = {"model": "native", "provider": provider, "max_tokens": 64, "effort": "low"}
    original = dict(selection)
    with pytest.raises(ValueError):
        providers.open_model(ModelSettings.from_selection(selection))
    assert selection == original


@pytest.mark.parametrize("model", [None, 42, True, [], {}])
def test_non_string_model_is_rejected_before_clients_without_coercion(monkeypatch, model):
    def forbidden(**kwargs):
        raise AssertionError("Invalid model reached SDK construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(providers, "AsyncAnthropic", forbidden)
    selection = {"model": model, "provider": "openai", "max_tokens": 64, "effort": "low"}
    original = dict(selection)
    with pytest.raises(ValueError):
        providers.open_model(ModelSettings.from_selection(selection))
    assert selection == original


def test_missing_model_is_rejected_before_clients(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Missing model reached SDK construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(providers, "AsyncAnthropic", forbidden)
    with pytest.raises(ValueError):
        providers.open_model(ModelSettings.from_selection({"provider": "openai", "max_tokens": 64, "effort": "low"}))


@pytest.mark.parametrize("model", ["gpt-medium", None])
def test_selection_owns_dict_subclass_data_before_reading_or_validation(model):
    class Selection(dict):
        def get(self, *args):
            pytest.fail("Selection invoked a caller-owned dictionary getter")

    selection = Selection(model=model)
    original = dict(selection)
    if model is None:
        with pytest.raises(ValueError):
            ModelSettings.from_selection(selection)
    else:
        applied = ModelSettings.from_selection(selection)
        assert (applied.model, applied.provider, applied.effort, applied.max_tokens) == (
            "gpt",
            "openai",
            "medium",
            128000,
        )
    assert dict(selection) == original
