"""
Tools for the Multi-Agent Research Platform.
"""

from .web_search import WebSearchTool
from .pdf_parser import PDFParser
from .citation_extractor import CitationExtractor

__all__ = [
    "WebSearchTool",
    "PDFParser",
    "CitationExtractor",
]

