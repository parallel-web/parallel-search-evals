import socket

import pytest
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "mock-api-key")

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network prohibited in offline harness tests")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
