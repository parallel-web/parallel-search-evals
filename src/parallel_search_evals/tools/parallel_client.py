"""Call the Parallel Python SDK and admit model-visible response payloads.

The caller closes the SDK client. Search adds its explicit mode while
Extract omits empty optional search queries; result and excerpt limits remain
service defaults. Known-source admission runs before fetching requested URLs.
The response stays a complete native JSON object apart from the explicitly
selected local source filter, so unknown metadata survives model replay.
Both the HTTP operation timeout and the surrounding tool deadline remain active.
"""

import asyncio
import os
from importlib.metadata import version
from typing import Any

import httpx
from parallel import APIConnectionError, AsyncParallel, DefaultAsyncHttpxClient

from parallel_search_evals import config
from parallel_search_evals.config import TOOL_SECONDS
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools.source_filter import filter_response, validate_extract_urls

MODES = config.PARALLEL_MODES


class ParallelClient:
    def __init__(
        self,
        mode: str = config.PARALLEL_DEFAULT_MODE,
        model: str = "",
        client: httpx.AsyncClient | None = None,
        *,
        filter_benchmarks: bool = config.FILTER_BENCHMARKS,
    ) -> None:
        if mode not in MODES:
            raise ValueError("Parallel mode must be turbo, fast, basic or advanced")
        self.mode = mode
        self.model = model
        self.filter_benchmarks = filter_benchmarks
        self.sdk = AsyncParallel(
            api_key=os.environ.get("PARALLEL_API_KEY", ""),
            base_url=config.PARALLEL_BASE_URL,
            default_headers={"user-agent": f"parallel-search-evals/{version('parallel-search-evals')}"},
            http_client=client or DefaultAsyncHttpxClient(follow_redirects=False),
            max_retries=0,
            timeout=config.PARALLEL_REQUEST_SECONDS,
        )

    def _request_payload(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Attach model and search mode while leaving excerpt controls to the API."""
        payload = {**arguments, "client_model": self.model}
        if endpoint == "search":
            payload["mode"] = self.mode
        elif not payload.get("search_queries"):
            payload.pop("search_queries", None)
        return payload

    async def request(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.filter_benchmarks:
            validate_extract_urls(arguments.get("urls", ()))
        payload = self._request_payload(endpoint, arguments)
        operation = getattr(self.sdk.with_raw_response, endpoint)
        async with asyncio.timeout(TOOL_SECONDS):
            try:
                response = await operation(**payload)
            except APIConnectionError as error:
                if error.__cause__ is not None and not isinstance(error.__cause__, httpx.HTTPError):
                    raise error.__cause__ from None
                raise
        result = _response_payload(response.http_response)
        return filter_response(result) if self.filter_benchmarks else result


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    """Decode successful Parallel responses without discarding unknown JSON fields."""
    result = response.json()
    if not isinstance(result, dict):
        raise NativeResponseError("Parallel response must be a JSON object")
    return result
