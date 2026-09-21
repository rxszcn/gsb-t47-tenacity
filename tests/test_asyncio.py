# Copyright 2016 Étienne Bersac
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
import unittest
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar
from unittest import mock

try:
    import trio
except ImportError:
    have_trio = False
else:
    have_trio = True

import pytest

import tenacity
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry,
    retry_if_exception,
    retry_if_result,
    stop_after_attempt,
)
from tenacity import asyncio as tasyncio
from tenacity.stop import stop_all, stop_any, stop_never
from tenacity.wait import wait_chain, wait_combine, wait_fixed

from .test_tenacity import (
    NoIOErrorAfterCount,
    NoneReturnUntilAfterCount,
    current_time_ms,
)

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


def asynctest(callable_: _F) -> Callable[..., Any]:
    @wraps(callable_)
    def wrapper(*a: Any, **kw: Any) -> Any:
        return asyncio.run(callable_(*a, **kw))

    return wrapper


def _make_async_stop(
    threshold: int,
) -> Callable[[RetryCallState], Coroutine[Any, Any, bool]]:
    async def _astop(retry_state: RetryCallState) -> bool:
        return retry_state.attempt_number >= threshold

    return _astop


async def _run_until_stop(stop: Any) -> int:
    attempts = 0

    async def _always_fails() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("x")

    retrying = AsyncRetrying(
        stop=stop,
        wait=tenacity.wait_none(),
        sleep=lambda s: asyncio.sleep(0),
    )
    with pytest.raises(RetryError):
        await retrying(_always_fails)
    return attempts


async def _async_function(thing: NoIOErrorAfterCount) -> Any:
    await asyncio.sleep(0.00001)
    return thing.go()


@retry
async def _retryable_coroutine(thing: NoIOErrorAfterCount) -> Any:
    await asyncio.sleep(0.00001)
    return thing.go()


@retry(stop=stop_after_attempt(2))
async def _retryable_coroutine_with_2_attempts(thing: NoIOErrorAfterCount) -> Any:
    await asyncio.sleep(0.00001)
    return thing.go()


class TestAsyncio(unittest.TestCase):
    @asynctest
    async def test_retry(self) -> None:
        thing = NoIOErrorAfterCount(5)
        await _retryable_coroutine(thing)
        assert thing.counter == thing.count

    @asynctest
    async def test_wait_falsy_values_mean_no_wait(self) -> None:
        # Mirrors the sync test: falsy `wait` values reach AsyncRetrying from
        # untyped callers and must not raise from inside iter().
        for wait in (None, 0):
            thing = NoIOErrorAfterCount(2)
            retrying = AsyncRetrying(
                wait=wait,  # type: ignore[arg-type]
                stop=stop_after_attempt(5),
            )
            await retrying(_async_function, thing)
            assert thing.counter == thing.count

    @asynctest
    async def test_iscoroutinefunction(self) -> None:
        assert inspect.iscoroutinefunction(_retryable_coroutine)

    @asynctest
    async def test_retry_using_async_retying(self) -> None:
        thing = NoIOErrorAfterCount(5)
        retrying = AsyncRetrying()
        await retrying(_async_function, thing)
        assert thing.counter == thing.count

    @asynctest
    async def test_stop_after_attempt(self) -> None:
        thing = NoIOErrorAfterCount(2)
        try:
            await _retryable_coroutine_with_2_attempts(thing)
        except RetryError:
            assert thing.counter == 2

    def test_repr(self) -> None:
        repr(tasyncio.AsyncRetrying())

    def test_retry_attributes(self) -> None:
        assert hasattr(_retryable_coroutine, "retry")
        assert hasattr(_retryable_coroutine, "retry_with")

    @asynctest
    async def test_statistics_visible_through_outer_decorator(self) -> None:
        """Statistics must resolve when @retry is wrapped by another decorator.

        A well-behaved outer decorator uses functools.wraps, which copies the
        inner wrapper's ``__dict__`` (including ``statistics``). Rebinding the
        attribute on each call left the outer wrapper pointing at a stale empty
        dict. See issue #519.
        """

        def outer(fn: _F) -> _F:
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        @outer
        @retry(stop=stop_after_attempt(3))
        async def my_call() -> str:
            return "ok"

        assert await my_call() == "ok"
        assert my_call.statistics["attempt_number"] == 1
        assert my_call.statistics is my_call.__wrapped__.statistics

    def test_retry_preserves_argument_defaults(self) -> None:
        async def function_with_defaults(a: int = 1) -> int:
            return a

        async def function_with_kwdefaults(*, a: int = 1) -> int:
            return a

        retrying = AsyncRetrying(
            wait=tenacity.wait_fixed(0.01), stop=tenacity.stop_after_attempt(3)
        )
        wrapped_defaults_function = retrying.wraps(function_with_defaults)
        wrapped_kwdefaults_function = retrying.wraps(function_with_kwdefaults)

        self.assertEqual(
            function_with_defaults.__defaults__,
            wrapped_defaults_function.__defaults__,  # type: ignore[attr-defined]
        )
        self.assertEqual(
            function_with_kwdefaults.__kwdefaults__,
            wrapped_kwdefaults_function.__kwdefaults__,  # type: ignore[attr-defined]
        )

    @asynctest
    async def test_attempt_number_is_correct_for_interleaved_coroutines(self) -> None:
        attempts: list[Any] = []

        def after(retry_state: RetryCallState) -> None:
            attempts.append((retry_state.args[0], retry_state.attempt_number))

        thing1 = NoIOErrorAfterCount(3)
        thing2 = NoIOErrorAfterCount(3)

        await asyncio.gather(
            _retryable_coroutine.retry_with(after=after)(thing1),
            _retryable_coroutine.retry_with(after=after)(thing2),
        )

        # There's no waiting on retry, only a wait in the coroutine, so the
        # executions should be interleaved.
        even_thing_attempts = attempts[::2]
        things, attempt_nos1 = zip(*even_thing_attempts)
        assert len(set(things)) == 1
        assert list(attempt_nos1) == [1, 2, 3]

        odd_thing_attempts = attempts[1::2]
        things, attempt_nos2 = zip(*odd_thing_attempts)
        assert len(set(things)) == 1
        assert list(attempt_nos2) == [1, 2, 3]


class TestAsyncStopCombinators(unittest.TestCase):
    @asynctest
    async def test_bare_async_stop(self) -> None:
        assert await _run_until_stop(_make_async_stop(4)) == 4

    @asynctest
    async def test_stop_any_with_async_member(self) -> None:
        assert await _run_until_stop(stop_any(_make_async_stop(4))) == 4

    @asynctest
    async def test_stop_any_with_async_and_sync_members(self) -> None:
        stop = stop_any(_make_async_stop(4), stop_after_attempt(99))
        assert await _run_until_stop(stop) == 4

    @asynctest
    async def test_stop_any_with_async_member_and_stop_never(self) -> None:
        assert await _run_until_stop(stop_any(_make_async_stop(4), stop_never)) == 4

    @asynctest
    async def test_stop_all_with_async_and_sync_members(self) -> None:
        # Both conditions must hold: the async one first fires at attempt 4,
        # the sync one is already true by then, so this stops at 4. If the
        # coroutine were misread as truthy it would stop at 2 instead.
        stop = stop_all(_make_async_stop(4), stop_after_attempt(2))
        assert await _run_until_stop(stop) == 4

    @asynctest
    async def test_stop_all_with_two_async_members(self) -> None:
        stop = stop_all(_make_async_stop(2), _make_async_stop(4))
        assert await _run_until_stop(stop) == 4

    @asynctest
    async def test_or_operator_with_async_member(self) -> None:
        assert await _run_until_stop(stop_after_attempt(99) | _make_async_stop(4)) == 4
        assert await _run_until_stop(_make_async_stop(4) | stop_after_attempt(99)) == 4

    @asynctest
    async def test_and_operator_with_async_member(self) -> None:
        assert await _run_until_stop(stop_after_attempt(2) & _make_async_stop(4)) == 4
        assert await _run_until_stop(_make_async_stop(4) & stop_after_attempt(2)) == 4

    def test_sync_only_combinators_stay_sync(self) -> None:
        from tenacity.stop import stop_any as sync_stop_any

        stop = stop_any(stop_after_attempt(4), stop_after_attempt(99))
        assert type(stop) is sync_stop_any
        assert not inspect.iscoroutinefunction(stop.__call__)

    def test_async_combinator_is_coroutine_callable(self) -> None:
        stop = stop_any(_make_async_stop(4))
        assert inspect.iscoroutinefunction(stop.__call__)
        assert isinstance(stop, tasyncio.stop_any)

    def test_async_namespace_exports_stop_and_wait(self) -> None:
        for name in ("stop_any", "stop_all", "wait_combine", "wait_chain"):
            assert hasattr(tasyncio, name), name


class TestAsyncWaitCombinators(unittest.TestCase):
    @staticmethod
    def _make_async_wait(
        value: float,
    ) -> Callable[[RetryCallState], Coroutine[Any, Any, float]]:
        async def _await_value(retry_state: RetryCallState) -> float:
            return value

        return _await_value

    async def _recorded_sleeps(self, wait: Any) -> list:
        sleeps = []

        def _sleep(seconds: float) -> Coroutine[Any, Any, None]:
            sleeps.append(seconds)
            return asyncio.sleep(0)

        calls = 0

        async def _fail_twice() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("x")
            return "ok"

        retrying = AsyncRetrying(stop=stop_after_attempt(3), wait=wait, sleep=_sleep)
        assert await retrying(_fail_twice) == "ok"
        return sleeps

    @asynctest
    async def test_bare_async_wait(self) -> None:
        assert await self._recorded_sleeps(self._make_async_wait(1.5)) == [1.5, 1.5]

    @asynctest
    async def test_wait_combine_with_async_member(self) -> None:
        wait = wait_combine(wait_fixed(1), self._make_async_wait(1.5))
        assert await self._recorded_sleeps(wait) == [2.5, 2.5]

    @asynctest
    async def test_wait_add_with_async_member(self) -> None:
        wait = wait_fixed(1) + self._make_async_wait(1.5)
        assert await self._recorded_sleeps(wait) == [2.5, 2.5]

    @asynctest
    async def test_wait_chain_with_async_member(self) -> None:
        wait = wait_chain(self._make_async_wait(1.5), wait_fixed(2))
        assert await self._recorded_sleeps(wait) == [1.5, 2.0]

    @asynctest
    async def test_async_wait_base_add(self) -> None:
        class _async_wait(tasyncio.async_wait_base):
            async def __call__(self, retry_state: RetryCallState) -> float:
                return 1.5

        wait = _async_wait() + wait_fixed(1)
        assert await self._recorded_sleeps(wait) == [2.5, 2.5]

    def test_sync_only_wait_combine_stays_sync(self) -> None:
        wait = wait_combine(wait_fixed(1), wait_fixed(2))
        assert not inspect.iscoroutinefunction(wait.__call__)
        assert type(wait) is tenacity.wait.wait_combine


class TestAsyncEnabled(unittest.TestCase):
    @asynctest
    async def test_enabled_false_skips_retry(self) -> None:
        """When enabled=False, async function is called directly without retrying."""
        call_count = 0

        @retry(enabled=False, stop=stop_after_attempt(3))
        async def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await always_fails()
        assert call_count == 1

    @asynctest
    async def test_enabled_false_aiter_raises_original_exception(self) -> None:
        """When enabled=False, the async iterator raises the original exception,
        not a RetryError, and the body executes exactly once."""
        call_count = 0
        retrying = AsyncRetrying(
            enabled=False,
            stop=stop_after_attempt(5),
        )
        with pytest.raises(ValueError, match="fail"):
            async for attempt in retrying:
                with attempt:
                    call_count += 1
                    raise ValueError("fail")
        assert call_count == 1

    @asynctest
    async def test_enabled_false_aiter_succeeds_on_first_attempt(self) -> None:
        """When enabled=False, the async iterator runs the body once and stops."""
        call_count = 0
        retrying = AsyncRetrying(
            enabled=False,
            stop=stop_after_attempt(5),
        )
        async for attempt in retrying:
            with attempt:
                call_count += 1
        assert call_count == 1

    @asynctest
    async def test_enabled_false_call_raises_original_exception(self) -> None:
        """When enabled=False, awaiting the controller directly raises the original
        exception, not a RetryError, and the coroutine executes exactly once."""
        call_count = 0

        async def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        retrying = AsyncRetrying(
            enabled=False,
            stop=stop_after_attempt(5),
        )
        with pytest.raises(ValueError, match="fail"):
            await retrying(always_fails)
        assert call_count == 1

    @asynctest
    async def test_enabled_false_call_succeeds_on_first_attempt(self) -> None:
        """When enabled=False, awaiting the controller directly runs the coroutine
        once and returns its result."""
        call_count = 0

        async def succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        retrying = AsyncRetrying(
            enabled=False,
            stop=stop_after_attempt(5),
        )
        assert await retrying(succeeds) == "ok"
        assert call_count == 1


@unittest.skipIf(not have_trio, "trio not installed")
class TestTrio(unittest.TestCase):
    def test_trio_basic(self) -> None:
        thing = NoIOErrorAfterCount(5)

        @retry
        async def trio_function() -> Any:
            await trio.sleep(0.00001)
            return thing.go()

        trio.run(trio_function)

        assert thing.counter == thing.count


class TestContextManager(unittest.TestCase):
    @asynctest
    async def test_do_max_attempts(self) -> None:
        attempts = 0
        retrying = tasyncio.AsyncRetrying(stop=stop_after_attempt(3))
        try:
            async for attempt in retrying:
                with attempt:
                    attempts += 1
                    raise Exception
        except RetryError:
            pass

        assert attempts == 3

    @asynctest
    async def test_async_with_attempt_manager(self) -> None:
        """AttemptManager supports async with for use inside async for."""
        attempts = 0
        retrying = tasyncio.AsyncRetrying(stop=stop_after_attempt(3))
        try:
            async for attempt in retrying:
                async with attempt:
                    attempts += 1
                    raise Exception
        except RetryError:
            pass

        assert attempts == 3

    @asynctest
    async def test_reraise(self) -> None:
        class CustomError(Exception):
            pass

        try:
            async for attempt in tasyncio.AsyncRetrying(
                stop=stop_after_attempt(1), reraise=True
            ):
                with attempt:
                    raise CustomError
        except CustomError:
            pass
        else:
            raise Exception

    @asynctest
    async def test_reraise_try_again_with_cause(self) -> None:
        # When TryAgain is raised from within an "except" block, reraise=True
        # should surface the underlying exception rather than TryAgain itself.
        class UnderlyingError(Exception):
            pass

        async def _test() -> None:
            try:
                raise UnderlyingError("boom")
            except UnderlyingError:
                # Implicit chaining via __context__ is exactly what we test.
                raise tenacity.TryAgain  # noqa: B904

        retrying = tasyncio.AsyncRetrying(
            stop=stop_after_attempt(2),
            retry=tenacity.retry_never,
            reraise=True,
        )
        with pytest.raises(UnderlyingError):
            await retrying(_test)
        self.assertEqual(2, retrying.statistics["attempt_number"])

    @asynctest
    async def test_sleeps(self) -> None:
        start = current_time_ms()
        try:
            async for attempt in tasyncio.AsyncRetrying(
                stop=stop_after_attempt(1), wait=wait_fixed(1)
            ):
                with attempt:
                    raise Exception
        except RetryError:
            pass
        t = current_time_ms() - start
        self.assertLess(t, 1.1)

    @asynctest
    async def test_retry_with_result(self) -> None:
        async def test() -> int:
            attempts = 0

            # mypy doesn't have great lambda support
            def lt_3(x: float) -> bool:
                return x < 3

            async for attempt in tasyncio.AsyncRetrying(retry=retry_if_result(lt_3)):
                with attempt:
                    attempts += 1
                attempt.retry_state.set_result(attempts)
            return attempts

        result = await test()

        self.assertEqual(3, result)

    @asynctest
    async def test_retry_with_async_result(self) -> None:
        async def test() -> int:
            attempts = 0

            async def lt_3(x: float) -> bool:
                return x < 3

            async for attempt in tasyncio.AsyncRetrying(
                retry=tasyncio.retry_if_result(lt_3)
            ):
                with attempt:
                    attempts += 1

                assert attempt.retry_state.outcome  # help mypy
                if not attempt.retry_state.outcome.failed:
                    attempt.retry_state.set_result(attempts)

            return attempts

        result = await test()

        self.assertEqual(3, result)

    @asynctest
    async def test_retry_with_async_exc(self) -> None:
        async def test() -> int:
            attempts = 0

            class CustomException(Exception):
                pass

            async def is_exc(e: BaseException) -> bool:
                return isinstance(e, CustomException)

            async for attempt in tasyncio.AsyncRetrying(
                retry=tasyncio.retry_if_exception(is_exc)
            ):
                with attempt:
                    attempts += 1
                    if attempts < 3:
                        raise CustomException

                assert attempt.retry_state.outcome  # help mypy
                if not attempt.retry_state.outcome.failed:
                    attempt.retry_state.set_result(attempts)

            return attempts

        result = await test()

        self.assertEqual(3, result)

    @asynctest
    async def test_retry_with_async_result_or(self) -> None:
        async def test() -> int:
            attempts = 0

            async def lt_3(x: float) -> bool:
                return x < 3

            class CustomException(Exception):
                pass

            def is_exc(e: BaseException) -> bool:
                return isinstance(e, CustomException)

            retry_strategy = tasyncio.retry_if_result(lt_3) | retry_if_exception(is_exc)
            async for attempt in tasyncio.AsyncRetrying(retry=retry_strategy):
                with attempt:
                    attempts += 1
                    if 2 < attempts < 4:
                        raise CustomException

                assert attempt.retry_state.outcome  # help mypy
                if not attempt.retry_state.outcome.failed:
                    attempt.retry_state.set_result(attempts)

            return attempts

        result = await test()

        self.assertEqual(4, result)

    @asynctest
    async def test_retry_with_async_result_ror(self) -> None:
        async def test() -> int:
            attempts = 0

            def lt_3(x: float) -> bool:
                return x < 3

            class CustomException(Exception):
                pass

            async def is_exc(e: BaseException) -> bool:
                return isinstance(e, CustomException)

            retry_strategy = retry_if_result(lt_3) | tasyncio.retry_if_exception(is_exc)
            async for attempt in tasyncio.AsyncRetrying(retry=retry_strategy):
                with attempt:
                    attempts += 1
                    if 2 < attempts < 4:
                        raise CustomException

                assert attempt.retry_state.outcome  # help mypy
                if not attempt.retry_state.outcome.failed:
                    attempt.retry_state.set_result(attempts)

            return attempts

        result = await test()

        self.assertEqual(4, result)

    @asynctest
    async def test_retry_with_async_result_and(self) -> None:
        async def test() -> int:
            attempts = 0

            async def lt_3(x: float) -> bool:
                return x < 3

            def gt_0(x: float) -> bool:
                return x > 0

            retry_strategy = tasyncio.retry_if_result(lt_3) & retry_if_result(gt_0)
            async for attempt in tasyncio.AsyncRetrying(retry=retry_strategy):
                with attempt:
                    attempts += 1
                attempt.retry_state.set_result(attempts)

            return attempts

        result = await test()

        self.assertEqual(3, result)

    @asynctest
    async def test_retry_with_async_result_rand(self) -> None:
        async def test() -> int:
            attempts = 0

            async def lt_3(x: float) -> bool:
                return x < 3

            def gt_0(x: float) -> bool:
                return x > 0

            retry_strategy = retry_if_result(gt_0) & tasyncio.retry_if_result(lt_3)
            async for attempt in tasyncio.AsyncRetrying(retry=retry_strategy):
                with attempt:
                    attempts += 1
                attempt.retry_state.set_result(attempts)

            return attempts

        result = await test()

        self.assertEqual(3, result)

    @asynctest
    async def test_async_retying_iterator(self) -> None:
        thing = NoIOErrorAfterCount(5)
        with pytest.raises(TypeError):
            for attempts in AsyncRetrying():
                with attempts:
                    await _async_function(thing)


class TestDecoratorWrapper(unittest.TestCase):
    @asynctest
    async def test_retry_function_attributes(self) -> None:
        """Test that the wrapped function attributes are exposed as intended.

        - statistics contains the value for the latest function run
        - retry object can be modified to change its behaviour (useful to patch in tests)
        - retry object statistics are synced with function statistics
        """

        self.assertTrue(
            await _retryable_coroutine_with_2_attempts(NoIOErrorAfterCount(1))
        )

        expected_stats = {
            "attempt_number": 2,
            "delay_since_first_attempt": mock.ANY,
            "idle_for": mock.ANY,
            "start_time": mock.ANY,
        }
        self.assertEqual(
            _retryable_coroutine_with_2_attempts.statistics,
            expected_stats,
        )
        self.assertEqual(
            _retryable_coroutine_with_2_attempts.retry.statistics,
            expected_stats,
        )

        with mock.patch.object(
            _retryable_coroutine_with_2_attempts.retry,
            "stop",
            tenacity.stop_after_attempt(1),
        ):
            try:
                self.assertTrue(
                    await _retryable_coroutine_with_2_attempts(NoIOErrorAfterCount(2))
                )
            except RetryError as exc:
                expected_stats = {
                    "attempt_number": 1,
                    "delay_since_first_attempt": mock.ANY,
                    "idle_for": mock.ANY,
                    "start_time": mock.ANY,
                }
                self.assertEqual(
                    _retryable_coroutine_with_2_attempts.statistics,
                    expected_stats,
                )
                self.assertEqual(exc.last_attempt.attempt_number, 1)
                self.assertEqual(
                    _retryable_coroutine_with_2_attempts.retry.statistics,
                    expected_stats,
                )
            else:
                self.fail("RetryError should have been raised after 1 attempt")


# make sure mypy accepts passing an async sleep function
# https://github.com/jd/tenacity/issues/399
async def my_async_sleep(x: float) -> None:
    await asyncio.sleep(x)


@retry(sleep=my_async_sleep)
async def foo() -> None:
    pass


class TestSyncFunctionWithAsyncSleep(unittest.TestCase):
    @asynctest
    async def test_sync_function_with_async_sleep(self) -> None:
        """A sync function with an async sleep callable uses AsyncRetrying."""
        mock_sleep = mock.AsyncMock()

        thing = NoneReturnUntilAfterCount(2)

        @retry(
            sleep=mock_sleep,
            wait=wait_fixed(1),
            retry=retry_if_result(lambda x: x is None),
        )
        def sync_function() -> Any:
            return thing.go()

        result = await sync_function()
        assert result is True
        assert mock_sleep.await_count == 2


if __name__ == "__main__":
    unittest.main()
