from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedDocument:
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class DocumentParser(ABC):
    name: str

    @abstractmethod
    async def parse(self, content: bytes, filename: str, content_type: str) -> ParsedDocument:
        raise NotImplementedError

