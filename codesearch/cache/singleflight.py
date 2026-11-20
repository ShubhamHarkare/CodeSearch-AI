import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class SingleFlight(Generic[T]):
    """Run at most one call per key at a time; concurrent callers share its result."""

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[T]] = {}

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Return (result, shared). shared=False for the leader, True for followers."""

        if key in self._inflight:
            # TODO: This is the follower so we need to wait for the first function to complete
            # TODO: We need to wait for that result to get complete and then return that result

            result: T = await asyncio.shield(self._inflight[key])

            return result, True
        else:
            # TODO: This means that there are no questions regarding this so we need to make this as
            # the leader and then move ahead

            future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
            self._inflight[key] = future

            try:
                result = await fn()
            except asyncio.CancelledError:
                future.cancel()
                raise
            except Exception as e:
                future.set_exception(e)
                future.exception()
                raise

            else:
                future.set_result(result)
                return result, False

            finally:
                self._inflight.pop(key, None)

    def __len__(self) -> int:
        """How many keys are in flight right now (0 when idle)."""
        return len(self._inflight)
