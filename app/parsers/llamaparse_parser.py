from app.parsers.base import DocumentParser, ParsedDocument


class LlamaParseParser(DocumentParser):
    name = "llamaparse"

    async def parse(self, content: bytes, filename: str, content_type: str) -> ParsedDocument:
        raise NotImplementedError(
            "Install and configure LlamaParse here without changing callers of DocumentParser."
        )

