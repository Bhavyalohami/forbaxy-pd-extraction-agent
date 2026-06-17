from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ReviewedExample(BaseModel):
    example_id: str = Field(default_factory=lambda: str(uuid4()))
    source_hash: str
    extracted_output: dict[str, Any]
    reviewed_output: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewLearningStore(ABC):
    @abstractmethod
    async def add_reviewed_example(self, example: ReviewedExample) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_recent_examples(self, limit: int = 10) -> list[ReviewedExample]:
        raise NotImplementedError


class InMemoryReviewLearningStore(ReviewLearningStore):
    def __init__(self) -> None:
        self._examples: list[ReviewedExample] = []

    async def add_reviewed_example(self, example: ReviewedExample) -> None:
        self._examples.append(example)

    async def list_recent_examples(self, limit: int = 10) -> list[ReviewedExample]:
        return self._examples[-limit:]

