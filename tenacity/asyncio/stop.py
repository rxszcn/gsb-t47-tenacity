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
from tenacity.stop import stop_base

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState
    from tenacity.stop import StopBaseT as _SyncStopBaseT


class async_stop_base(stop_base):
    """Abstract base class for async stop strategies."""

    @abc.abstractmethod
    @override
    async def __call__(  # type: ignore[override]
        self, retry_state: RetryCallState
    ) -> bool:
        pass


AsyncStopBaseT = (
    async_stop_base | typing.Callable[["RetryCallState"], typing.Awaitable[bool]]
)


class stop_any(async_stop_base):
    """Stop if any of the stop conditions is valid.

    Each condition may be a regular or an async callable; an async one is
    awaited instead of being judged for its truthiness as a coroutine object.
    """

    def __init__(self, *stops: async_stop_base | _SyncStopBaseT) -> None:
        self.stops = stops

    @override
    async def __call__(  # type: ignore[override]
        self, retry_state: RetryCallState
    ) -> bool:
        for stop in self.stops:
            if await _utils.wrap_to_async_func(stop)(retry_state):
                return True
        return False


class stop_all(async_stop_base):
    """Stop if all the stop conditions are valid.

    Each condition may be a regular or an async callable.
    """

    def __init__(self, *stops: async_stop_base | _SyncStopBaseT) -> None:
        self.stops = stops

    @override
    async def __call__(  # type: ignore[override]
        self, retry_state: RetryCallState
    ) -> bool:
        for stop in self.stops:
            if not await _utils.wrap_to_async_func(stop)(retry_state):
                return False
        return True
