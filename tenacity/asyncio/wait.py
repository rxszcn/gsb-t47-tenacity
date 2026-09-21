# Copyright 2016–2021 Julien Danjou
# Copyright 2016 Joshua Harlow
# Copyright 2013-2014 Ray Holder
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

import abc
import typing

from tenacity import _utils
from tenacity._utils import override
from tenacity.wait import wait_base

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState
    from tenacity.wait import WaitBaseT as SyncWaitBaseT


class async_wait_base(wait_base):
    """Abstract base class for async wait strategies."""

    @abc.abstractmethod
    @override
    async def __call__(self, retry_state: "RetryCallState") -> float:  # type: ignore[override]
        pass

    @override
    def __add__(  # type: ignore[override]
        self, other: "wait_base | async_wait_base"
    ) -> "wait_combine":
        return wait_combine(self, other)

    # `__radd__` is inherited from `wait_base`: it already routes through
    # `wait_combine`, whose factory upgrades to the async combinator when any
    # member is a coroutine callable.


WaitBaseT = (
    async_wait_base | typing.Callable[["RetryCallState"], typing.Awaitable[float]]
)


class wait_combine(async_wait_base):
    """Combine several waiting strategies."""

    def __init__(self, *strategies: "SyncWaitBaseT | WaitBaseT") -> None:
        self.wait_funcs = strategies

    @override
    async def __call__(self, retry_state: "RetryCallState") -> float:  # type: ignore[override]
        result = 0.0
        for wait_func in self.wait_funcs:
            result += await _utils.wrap_to_async_func(wait_func)(retry_state)
        return float(result)


class wait_chain(async_wait_base):
    """Chain two or more waiting strategies.

    If all strategies are exhausted, the very last strategy is used
    thereafter.
    """

    def __init__(self, *strategies: "SyncWaitBaseT | WaitBaseT") -> None:
        if not strategies:
            raise ValueError("wait_chain() requires at least one strategy")
        self.strategies = strategies

    @override
    async def __call__(self, retry_state: "RetryCallState") -> float:  # type: ignore[override]
        wait_func_no = min(max(retry_state.attempt_number, 1), len(self.strategies))
        wait_func = self.strategies[wait_func_no - 1]
        return typing.cast(
            "float", await _utils.wrap_to_async_func(wait_func)(retry_state)
        )
