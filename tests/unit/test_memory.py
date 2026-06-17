import pytest
from app.core.exceptions import NotFoundError
from app.models.chat import ChatMessage
from app.services.memory import InMemorySessionStore


@pytest.mark.asyncio
async def test_session_store_appends_and_reads_messages():
    store = InMemorySessionStore()
    await store.append("s1", ChatMessage(role="user", content="hello"))

    session = await store.get("s1")
    assert session.session_id == "s1"
    assert session.messages[0].content == "hello"


@pytest.mark.asyncio
async def test_session_store_delete_missing_raises():
    store = InMemorySessionStore()
    with pytest.raises(NotFoundError):
        await store.delete("missing")

