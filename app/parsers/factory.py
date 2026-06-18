from app.config.settings import Settings
from app.parsers.base import DocumentParser
from app.parsers.llamaparse_parser import LlamaParseParser
from app.parsers.mock_parser import MockParser


def build_parser(settings: Settings) -> DocumentParser:
    if settings.parser_mode == "llamaparse":
        return LlamaParseParser(api_key=settings.llama_cloud_api_key)
    return MockParser()
