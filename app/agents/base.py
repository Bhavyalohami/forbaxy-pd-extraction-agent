from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.schemas.pd import LearningContext, LearningMetadata


@dataclass(slots=True)
class AgentResult:
    response: str
    sources: list[dict[str, str]] = field(default_factory=list)
    learning_metadata: dict[str, int | float | str | bool | None] = field(default_factory=dict)


class AgentPort(ABC):
    @abstractmethod
    async def run(
        self,
        session_id: str,
        message: str,
        *,
        learning_context: LearningContext | None = None,
        learning_metadata: LearningMetadata | None = None,
        extraction_id: str | None = None,
    ) -> AgentResult:
        raise NotImplementedError
