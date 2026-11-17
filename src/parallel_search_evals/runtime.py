"""Shared failure identities and asynchronous task lifetime handling."""

import asyncio


class NativeResponseError(ValueError):
    """Native model output cannot support continuation or a final answer."""


class GradingError(ValueError):
    """The judge returned no valid official grade."""


async def join_tasks[T](tasks: list[asyncio.Task[T]]) -> list[T]:
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
