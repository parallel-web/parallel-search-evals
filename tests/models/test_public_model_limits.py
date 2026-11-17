import json

import google.auth
import httpx
import pytest

from parallel_search_evals.models import ModelSettings, open_model
from parallel_search_evals.tools import ToolDispatcher
from tests.models.test_gemini_native import Credentials
from tests.sdk_transport import mock_owned_transport


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("vertex", "gemini-4-carbon", 1048576),
        ("vertex", "gemini-4-hydrogen", 1048576),
        ("vertex", "gemini-4-argon", 1048576),
        ("vertex", "gemini-3.8-flash", 65536),
        ("vertex", "gemini-fixture", 65536),
        ("gemini", "gemini-4-carbon", 65536),
        ("openrouter", "moonshotai/kimi-k3.1-pro", 943718),
        ("openrouter", "tencent/hy4", 65536),
        ("openai", "gpt-6.1-sol", 128000),
        ("anthropic", "claude-opus-5-5", 128000),
    ],
)
def test_public_model_defaults_are_exact_model_and_route_specific(provider, model, expected):
    model_settings = ModelSettings.from_selection({"provider": provider, "model": model})
    assert model_settings.max_tokens == expected
    assert ModelSettings.model_validate(model_settings.model_dump()).max_tokens == expected


@pytest.mark.parametrize("provider,model", [("vertex", "gemini-4-carbon"), ("openrouter", "moonshotai/kimi-k3.1-pro")])
def test_explicit_cap_remains_an_explicit_override(provider, model):
    model_settings = ModelSettings.from_selection({"provider": provider, "model": model, "max_tokens": 1234})
    assert model_settings.max_tokens == 1234


@pytest.mark.parametrize("model", ["gemini-4-carbon", "gemini-4-hydrogen", "gemini-4-argon"])
async def test_owned_vertex_sdk_sends_public_output_ceiling_on_long_decode_api(monkeypatch, model):
    sent = []

    async def serve(request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "answer"}]}}]
            },
        )

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))
    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (Credentials(), "adc-project"))
    model_client = open_model(
        ModelSettings.from_selection(
            {"provider": "vertex", "model": model, "vertex_project": "project", "vertex_location": "global"}
        )
    )
    try:
        answer = await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert answer.text == "answer"
        assert len(sent) == 1
        assert str(sent[0].url) == (
            f"https://aiplatform.googleapis.com/v1beta1/projects/project/locations/global/publishers/google/models/{model}:generateContent"
        )
        assert json.loads(sent[0].content)["generationConfig"]["maxOutputTokens"] == 1048576
    finally:
        await model_client.aclose()


@pytest.mark.parametrize("backends", [("river",), ["river"]])
def test_river_specific_public_max_follows_validated_backend_tuple(backends):
    model_settings = ModelSettings.from_selection(
        {"provider": "openrouter", "model": "moonshotai/kimi-k3.1-pro", "openrouter_backends": backends}
    )
    assert model_settings.openrouter_backends == ("river",)
    assert model_settings.max_tokens == 1048575


@pytest.mark.parametrize(
    "backends", [("moonshotai/mxfp4",), ("moonshotai/mxfp4", "river"), ("river", "moonshotai/mxfp4"), ()]
)
def test_kimi_other_backend_selections_keep_conservative_model_default(backends):
    model_settings = ModelSettings.from_selection(
        {"provider": "openrouter", "model": "moonshotai/kimi-k3.1-pro", "openrouter_backends": backends}
    )
    assert model_settings.max_tokens == 943718


def test_explicit_river_cap_is_not_replaced_by_backend_default():
    model_settings = ModelSettings.from_selection(
        {
            "provider": "openrouter",
            "model": "moonshotai/kimi-k3.1-pro",
            "openrouter_backends": ["river"],
            "max_tokens": 1234,
        }
    )
    assert model_settings.max_tokens == 1234


@pytest.mark.parametrize("backends", [None, 42, [["river"]], {"river": True}])
def test_invalid_backend_shapes_still_raise_pydantic_validation_error(backends):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelSettings.from_selection(
            {"provider": "openrouter", "model": "moonshotai/kimi-k3.1-pro", "openrouter_backends": backends}
        )


def test_irrelevant_river_backend_option_does_not_change_vertex_model_cap():
    model_settings = ModelSettings.from_selection(
        {"provider": "vertex", "model": "gemini-4-carbon", "openrouter_backends": ["river"]}
    )
    assert model_settings.max_tokens == 1048576
