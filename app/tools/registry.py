from llama_index.core.tools import FunctionTool

from app.tools.base import ToolDefinition
from app.tools.builtin import (
    calculator,
    current_datetime,
    file_metadata_extractor,
    prescription_validator_placeholder,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def as_llamaindex_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(
                fn=definition.function,
                name=definition.name,
                description=definition.description,
            )
            for definition in self.list_tools()
        ]


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="current_datetime",
            description="Returns the current UTC date/time for follow-up reasoning.",
            function=current_datetime,
        )
    )
    registry.register(
        ToolDefinition(
            name="calculator",
            description="Safely evaluates arithmetic expressions.",
            function=calculator,
        )
    )
    registry.register(
        ToolDefinition(
            name="file_metadata_extractor",
            description="Extracts simple metadata from an uploaded PD-only crop.",
            function=file_metadata_extractor,
        )
    )
    registry.register(
        ToolDefinition(
            name="prescription_validator_placeholder",
            description="Placeholder validation hook for future prescription safety checks.",
            function=prescription_validator_placeholder,
        )
    )
    return registry
