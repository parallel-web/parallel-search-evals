"""A selected native route loads one exact credential without storing it."""

import json
from types import SimpleNamespace

import pytest

import parallel_search_evals.models as providers
from parallel_search_evals.models import gemini as google_module
from parallel_search_evals.models import settings as settings_module
from parallel_search_evals.models.settings import ModelSettings

ROUTES = [
    ("openai", "responses", "OPENAI_API_KEY"),
    ("anthropic", "messages", "ANTHROPIC_API_KEY"),
    ("openrouter", "responses", "OPENROUTER_API_KEY"),
    ("unbiased", "responses", "UNBIASED_API_KEY"),
    ("compatible", "responses", "CUSTOM_TEST_KEY"),
    ("compatible", "chat", "CUSTOM_TEST_KEY"),
    ("gemini", "contents", "GOOGLE_API_KEY"),
]


@pytest.mark.parametrize("provider,api,key_name", ROUTES)
@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_selected_missing_or_blank_key_fails_before_any_sdk(monkeypatch, provider, api, key_name, value):
    for name in (*settings_module.KEYS.values(), "CUSTOM_TEST_KEY", "UNUSED_KEY"):
        monkeypatch.setenv(name, "ambient-poison")
    if value is None:
        monkeypatch.delenv(key_name)
    else:
        monkeypatch.setenv(key_name, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid selected credential reached SDK construction")

    monkeypatch.setattr(providers, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(providers, "AsyncAnthropic", forbidden)
    monkeypatch.setattr(google_module.genai, "Client", forbidden)
    model_settings = ModelSettings.from_selection(
        {
            "model": "native-opaque",
            "provider": provider,
            "max_tokens": 64,
            "api": api,
            "base_url": "https://custom.example/v1",
            "openrouter_backends": ["backend"],
            "api_key_env": key_name if provider == "compatible" else "UNUSED_KEY",
        }
    )
    with pytest.raises(ValueError, match=key_name):
        providers.open_model(model_settings)


@pytest.mark.parametrize("provider,api,key_name", ROUTES)
def test_selected_key_is_read_once_returned_unstripped_and_not_persisted(monkeypatch, provider, api, key_name):
    reads, received = [], []
    original_key = "  selected-original-key  "

    def read(name, default=""):
        reads.append(name)
        return original_key if name == key_name else "ambient-poison"

    def sdk(**kwargs):
        received.append(kwargs["api_key"])
        return SimpleNamespace()

    def developer(*args, **kwargs):
        received.append(kwargs["api_key"])
        return SimpleNamespace()

    monkeypatch.setattr(settings_module, "os", SimpleNamespace(environ=SimpleNamespace(get=read)))
    monkeypatch.setattr(providers, "AsyncOpenAI", sdk)
    monkeypatch.setattr(providers, "AsyncAnthropic", sdk)
    monkeypatch.setattr(google_module.genai, "Client", developer)
    model_settings = ModelSettings.from_selection(
        {
            "model": "native-opaque",
            "provider": provider,
            "max_tokens": 64,
            "api": api,
            "base_url": "https://custom.example/v1",
            "openrouter_backends": ["backend"],
            "api_key_env": key_name if provider == "compatible" else "UNUSED_KEY",
        }
    )
    before = model_settings.model_dump()
    providers.open_model(model_settings)
    assert reads == [key_name]
    assert received == [original_key]
    assert model_settings.model_dump() == before
    assert "selected-original-key" not in json.dumps(model_settings.model_dump())


def test_vertex_never_loads_an_api_key(monkeypatch):
    calls = []

    def forbidden(self):
        raise AssertionError("Vertex attempted API-key loading")

    def vertex(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(ModelSettings, "load_api_key", forbidden)
    monkeypatch.setattr(google_module.genai, "Client", vertex)
    providers.open_model(
        ModelSettings.from_selection(
            {
                "model": "native-opaque",
                "provider": "vertex",
                "vertex_project": "project",
                "vertex_location": "global",
            }
        )
    )
    assert len(calls) == 1 and calls[0]["project"] == "project"
