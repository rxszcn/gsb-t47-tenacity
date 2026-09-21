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
from tenacity.stop import stop_base

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState
    from tenacity.stop import StopBaseT as SyncStopBaseT


class async_stop_base(stop_base):
    """Abstract base class for async stop strategies."""

    @abc.abstractmethod
    @override
    async def __call__(self, retry_state: "RetryCallState") -> bool:  # type: ignore[override]
        pass

    @override
    def __and__(  # type: ignore[override]
        self, other: "stop_base | async_stop_base"
    ) -> "stop_all":
        return stop_all(self, other)

    @override
    def __rand__(  # type: ignore[misc,override]
        self, other: "stop_base | async_stop_base"
    ) -> "stop_all":
        return stop_all(other, self)

    @override
    def __or__(  # type: ignore[override]
        self, other: "stop_base | async_stop_base"
    ) -> "stop_any":
        return stop_any(self, other)

    @override
    def __ror__(  # type: ignore[misc,override]
        self, other: "stop_base | async_stop_base"
    ) -> "stop_any":
        return stop_any(other, self)


StopBaseT = (
    async_stop_base | typing.Callable[["RetryCallState"], typing.Awaitable[bool]]
)


class stop_any(async_stop_base):
    """Stop if any of the stop conditions is valid."""

    def __init__(self, *stops: "SyncStopBaseT | StopBaseT") -> None:
        self.stops = stops

    @override
    async def __call__(self, retry_state: "RetryCallState") -> bool:  # type: ignore[override]
        result = False
        for stop in self.stops:
            result = result or await _utils.wrap_to_async_func(stop)(retry_state)
            if result:
                break
        return result


class stop_all(async_stop_base):
    """Stop if all the stop conditions are valid."""

    def __init__(self, *stops: "SyncStopBaseT | StopBaseT") -> None:
        self.stops = stops

    @override
    async def __call__(self, retry_state: "RetryCallState") -> bool:  # type: ignore[override]
        result = True
        for stop in self.stops:
            result = result and await _utils.wrap_to_async_func(stop)(retry_state)
            if not result:
                break
        return result
