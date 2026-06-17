from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class AgentResult:
    response: str
    sources: list[dict[str, str]] = field(default_factory=list)


class AgentPort(ABC):
    @abstractmethod
    async def run(self, session_id: str, message: str) -> AgentResult:
        raise NotImplementedError

