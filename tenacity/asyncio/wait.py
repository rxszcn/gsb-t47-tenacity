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
from __future__ import annotations

import abc
import typing

from tenacity import _utils
from tenacity._utils import override
from tenacity.wait import wait_base

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState
    from tenacity.wait import WaitBaseT as _SyncWaitBaseT


class async_wait_base(wait_base):
    """Abstract base class for async wait strategies."""

    @abc.abstractmethod
    @override
    async def __call__(  # type: ignore[override]
        self, retry_state: RetryCallState
    ) -> float:
        pass


AsyncWaitBaseT = (
    async_wait_base
    | typing.Callable[["RetryCallState"], typing.Awaitable[float | int] | float | int]
)


class wait_combine(async_wait_base):
    """Combine several waiting strategies, sync or async.

    The resulting wait is the sum of each member's value, awaiting async
    members before summing.
    """

    def __init__(self, *strategies: async_wait_base | _SyncWaitBaseT) -> None:
        self.wait_funcs = strategies

    @override
    async def __call__(  # type: ignore[override]
        self, retry_state: RetryCallState
    ) -> float:
        # Positional, like the synchronous wait_combine: a WaitBaseT callable
        # is only guaranteed to take the state positionally.
        total = 0.0
        for strategy in self.wait_funcs:
            total += await _utils.wrap_to_async_func(strategy)(retry_state)
        return total
