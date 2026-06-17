from app.parsers.base import DocumentParser, ParsedDocument


class MockParser(DocumentParser):
    name = "mock"

    async def parse(self, content: bytes, filename: str, content_type: str) -> ParsedDocument:
        preview = content[:1000].decode("utf-8", errors="ignore")
        return ParsedDocument(
            text=preview,
            metadata={
                "filename": filename,
                "content_type": content_type,
                "note": "Mock parser only. Replace with LlamaParseParser behind DocumentParser.",
            },
        )

