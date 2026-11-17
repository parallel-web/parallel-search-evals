from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

import parallel_search_evals.models as providers
from parallel_search_evals.models import settings as settings_module
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.tools import ToolDispatcher
from tests.sdk_transport import mock_owned_transport

BAD_ENDPOINTS = [
    "http://remote.example/v1",
    "http://192.168.1.1/v1",
    "http://169.254.169.254/v1",
    "http://0.0.0.0/v1",
    "http://localhost.evil.example/v1",
    "http://evil-localhost/v1",
    "http://localhost./v1",
    "http://localhost%2eevil.example/v1",
    "http://127.0.0.1.evil.example/v1",
    "http://127.1/v1",
    "http://2130706433/v1",
    "http://0177.0.0.1/v1",
    "http://[::]/v1",
    "http://[fd00::1]/v1",
    "http://[::ffff:8.8.8.8]/v1",
    "https://@remote.example/v1",
    "https://user:password@remote.example/v1",
    "http://user@localhost/v1",
    "https://remote.example/v1?key=invalid",
    "http://localhost/v1#fragment",
    "http://localhost:65536/v1",
    "http://localhost:invalid/v1",
    "http://[::1]evil.example/v1",
]
GOOD_ENDPOINTS = [
    "https://remote.example/v1",
    "https://remote.example:8443/v1",
    "http://localhost/v1",
    "http://LOCALHOST:8080/v1",
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.2/v1",
    "http://127.255.255.254:65535/v1",
    "http://[::1]:8000/v1",
    "http://[::ffff:127.0.0.1]/v1",
    "http://[0:0:0:0:0:0:0:1]/v1",
]


def sdk_fixture(monkeypatch, api):
    requests, clients, credential_reads = [], [], []

    def serve(request):
        requests.append(request)
        if api == "chat":
            return httpx.Response(
                200,
                json={
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture",
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
            },
        )

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))

    def sdk(**kwargs):
        client = AsyncOpenAI(**kwargs)
        clients.append(client)
        return client

    def read(name, default=""):
        credential_reads.append(name)
        return "fixture-key"

    monkeypatch.setattr(providers, "AsyncOpenAI", sdk)
    monkeypatch.setattr(settings_module, "os", SimpleNamespace(environ=SimpleNamespace(get=read)))
    return requests, clients, credential_reads


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("endpoint", BAD_ENDPOINTS)
async def test_bad_endpoint_rejected_before_credentials_and_actual_sdk_dispatch(monkeypatch, api, endpoint):
    requests, clients, reads = sdk_fixture(monkeypatch, api)
    model_settings = ModelSettings.from_selection(
        {"model": "fixture", "provider": "compatible", "base_url": endpoint, "api": api, "max_tokens": 64}
    )
    try:
        with pytest.raises(ValueError):
            model = providers.open_model(model_settings)
            await model.generate(model.new_history().append_question("question"), ToolDispatcher([]))
        assert requests == [] and reads == [] and clients == []
    finally:
        for client in clients:
            await client.close()


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("endpoint", GOOD_ENDPOINTS)
async def test_https_and_literal_loopback_actual_sdk_dispatch_preserves_endpoint(monkeypatch, api, endpoint):
    requests, clients, reads = sdk_fixture(monkeypatch, api)
    model_settings = ModelSettings.from_selection(
        {"model": "fixture", "provider": "compatible", "base_url": endpoint, "api": api, "max_tokens": 64}
    )
    before = model_settings.model_dump()
    try:
        model = providers.open_model(model_settings)
        output = await model.generate(model.new_history().append_question("question"), ToolDispatcher([]))
        assert output.text == "answer"
        assert len(requests) == 1 and requests[0].url.host == httpx.URL(endpoint).host
        assert requests[0].url.scheme == httpx.URL(endpoint).scheme
        assert reads == ["COMPATIBLE_API_KEY"]
        assert model_settings.model_dump() == before and model_settings.base_url == endpoint
    finally:
        for client in clients:
            await client.close()


@pytest.mark.parametrize(
    "name", ["", "not-a-valid-name", "1BAD", "KEY WITH SPACE", "KEY\n", "lowercase", "éKEY", "KEY=BAD"]
)
@pytest.mark.parametrize("api", ["responses", "chat"])
async def test_invalid_environment_name_rejected_before_lookup_or_actual_sdk(monkeypatch, api, name):
    requests, clients, reads = sdk_fixture(monkeypatch, api)
    try:
        with pytest.raises(ValueError):
            model_settings = ModelSettings.from_selection(
                {
                    "model": "fixture",
                    "provider": "compatible",
                    "base_url": "https://remote.example/v1",
                    "api": api,
                    "api_key_env": name,
                    "max_tokens": 64,
                }
            )
            model = providers.open_model(model_settings)
            await model.generate(model.new_history().append_question("question"), ToolDispatcher([]))
        assert reads == [] and clients == [] and requests == []
    finally:
        for client in clients:
            await client.close()


@pytest.mark.parametrize("name", [None, "CUSTOM_TEST_KEY", "_LOCAL_KEY", "KEY_123"])
async def test_valid_environment_name_reads_exact_selected_variable(monkeypatch, name):
    requests, clients, reads = sdk_fixture(monkeypatch, "responses")
    model_settings = ModelSettings.from_selection(
        {
            "model": "fixture",
            "provider": "compatible",
            "base_url": "http://127.0.0.1/v1",
            "api_key_env": name,
            "max_tokens": 64,
        }
    )
    try:
        model = providers.open_model(model_settings)
        assert (
            await model.generate(model.new_history().append_question("question"), ToolDispatcher([]))
        ).text == "answer"
        assert reads == [name if name is not None else "COMPATIBLE_API_KEY"] and len(requests) == 1
    finally:
        for client in clients:
            await client.close()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter", "unbiased", "gemini", "vertex"])
def test_irrelevant_compatible_controls_remain_ignored(provider):
    model_settings = ModelSettings.from_selection(
        {
            "model": "fixture",
            "provider": provider,
            "base_url": "http://remote.example/v1",
            "api_key_env": "UNUSED_POISON_KEY",
            "api": "unused",
            "max_tokens": 64,
            "openrouter_backends": ["backend"],
            "effort": "none",
        }
    )
    model_settings.validate_selected_route()
    assert model_settings.api_key_env == "UNUSED_POISON_KEY" and model_settings.base_url == "http://remote.example/v1"


@pytest.mark.parametrize(
    "provider", ["openai", "anthropic", "openrouter", "unbiased", "gemini", "vertex", "compatible"]
)
def test_invalid_name_is_rejected_without_echoing_literal_credential(provider):
    import traceback

    private = "sk-" + "S" * 40
    with pytest.raises(ValueError) as failure:
        ModelSettings.from_selection(
            {"model": "fixture", "provider": provider, "max_tokens": 64, "api_key_env": private}
        )
    assert private not in str(failure.value)
    assert private not in "".join(traceback.format_exception(failure.value))
