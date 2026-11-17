from functools import wraps

import httpx


def mock_owned_transport(monkeypatch, transport):
    original = httpx.AsyncClient.__init__

    @wraps(original)
    def init(self, *args, **kwargs):
        if not isinstance(kwargs.get("transport"), httpx.MockTransport):
            kwargs["transport"] = transport
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
