from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentLoaderError(Exception):
    """Raised when the knowledge base directory cannot be loaded."""


@dataclass(frozen=True)
class Page:
    """A single page of text extracted from a source document.

    PDFs are split page by page; TXT/MD files are treated as a single
    page so page-level provenance is always available.
    """

    text: str
    source: str
    page: int
    total_pages: int

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "total_pages": self.total_pages,
        }


@dataclass
class LoadReport:
    """Summary of a load operation, including problems found."""

    documents: int = 0
    pages: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _strip(text: str) -> str:
    return " ".join(text.split())


def _load_pdf(path: Path, report: LoadReport, pages: list[Page]) -> None:
    try:
        reader = PdfReader(str(path))
        total = len(reader.pages)
        for index, page in enumerate(reader.pages, start=1):
            text = _strip(page.extract_text() or "")
            if text:
                pages.append(Page(text=text, source=path.name, page=index, total_pages=total))
    except Exception as exc:
        report.errors.append(f"{path.name}: erro ao ler PDF ({exc})")


def _load_text(path: Path, pages: list[Page]) -> None:
    text = _strip(path.read_text(encoding="utf-8", errors="replace"))
    if text:
        pages.append(Page(text=text, source=path.name, page=1, total_pages=1))


def load_documents(directory: str | Path) -> tuple[list[Page], LoadReport]:
    """Load every supported document under ``directory``.

    Invalid or corrupt files are skipped individually, recorded in the
    report and never abort the whole load.

    Returns a tuple of ``(pages, report)``.
    """
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise DocumentLoaderError(f"diretorio de documentos nao encontrado: {root}")

    pages: list[Page] = []
    report = LoadReport()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or str(path.name).startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report.skipped += 1
            continue

        report.documents += 1
        if path.suffix.lower() == ".pdf":
            _load_pdf(path, report, pages)
        else:
            _load_text(path, pages)

    if not pages and not report.errors:
        report.errors.append("nenhum documento carregado (base de conhecimento vazia)")

    report.pages = len(pages)
    logger.info(
        "carga concluida: %d documento(s), %d pagina(s), %d ignorado(s)",
        report.documents,
        report.pages,
        report.skipped,
    )
    return pages, report
