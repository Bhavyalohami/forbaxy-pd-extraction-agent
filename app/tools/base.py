from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol


class ToolCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, str]:
        ...


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    function: Callable[..., Any]

