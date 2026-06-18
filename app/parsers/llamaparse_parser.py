import asyncio
import os
import tempfile
from pathlib import Path

from app.core.exceptions import AppError
from app.parsers.base import DocumentParser, ParsedDocument


class LlamaParseParser(DocumentParser):
    name = "llamaparse"

    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key.strip()

    async def parse(self, content: bytes, filename: str, content_type: str) -> ParsedDocument:
        if not self._api_key:
            raise AppError(
                "LLAMA_CLOUD_API_KEY is required when PARSER_MODE=llamaparse.",
                error_code="LLAMAPARSE_NOT_CONFIGURED",
                status_code=503,
            )

        return await asyncio.to_thread(self._parse_sync, content, filename, content_type)

    def _parse_sync(self, content: bytes, filename: str, content_type: str) -> ParsedDocument:
        try:
            from llama_cloud import LlamaCloud
        except ImportError as exc:
            raise AppError(
                "llama-cloud package is not installed. "
                "Run dependency sync before using LlamaParse.",
                error_code="LLAMAPARSE_DEPENDENCY_MISSING",
                status_code=503,
            ) from exc

        suffix = Path(filename).suffix or self._suffix_for_content_type(content_type)
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(content)
                temp_path = handle.name

            client = LlamaCloud(api_key=self._api_key)
            with open(temp_path, "rb") as file_handle:
                uploaded_file = client.files.create(
                    file=(Path(filename).name or "pd-crop", file_handle, content_type),
                    purpose="parse",
                )
            result = client.parsing.parse(
                file_id=uploaded_file.id,
                tier="agentic",
                version="latest",
                expand=["markdown"],
            )
            markdown = self._extract_markdown(result)
            if markdown.strip() == "":
                raise AppError(
                    "Unable to parse prescription details image.",
                    error_code="LLAMAPARSE_EMPTY_RESULT",
                    status_code=422,
                )

            return ParsedDocument(
                text=markdown,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    "parser": self.name,
                    "llamaparse_file_id": str(uploaded_file.id),
                },
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                "Unable to parse prescription details image.",
                error_code="LLAMAPARSE_FAILED",
                status_code=502,
            ) from exc
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _extract_markdown(self, result: object) -> str:
        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, str):
            return markdown

        pages = getattr(markdown, "pages", None)
        if pages:
            page_text = []
            for page in pages:
                page_markdown = getattr(page, "markdown", None)
                if isinstance(page_markdown, str) and page_markdown.strip():
                    page_text.append(page_markdown)
            return "\n\n".join(page_text)

        if isinstance(result, dict):
            full = result.get("markdown_full") or result.get("text_full")
            if isinstance(full, str):
                return full

        return ""

    def _suffix_for_content_type(self, content_type: str) -> str:
        mapping = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "application/pdf": ".pdf",
        }
        return mapping.get(content_type.lower(), ".bin")
