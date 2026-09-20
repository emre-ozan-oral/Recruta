"""Turn an uploaded CV / job-posting file into plain text.

Design goal: recover as much as is *actually recoverable* before giving up,
and never lie about what was recovered.

Pipeline per file:
  1. Sniff the real type from the bytes (magic numbers), not just the
     extension — a .pdf that is really a .docx, or a screenshot saved as
     .pdf, still gets parsed by the right extractor.
  2. PDF:  pypdf -> (empty-password decrypt) -> rebuild a damaged xref/trailer
           and retry -> pypdfium2 -> automatic OCR of the rendered pages
           (scanned PDFs have no text layer; that is not an error).
  3. DOCX: python-docx -> raw-XML fallback (also catches text boxes, which
           python-docx cannot see and CV templates love to use).
  4. Images: Groq vision OCR (model configurable via GROQ_VISION_MODEL).

Whatever was repaired/approximated is reported in ExtractionResult.notes so
the UI can tell the user ("recovered from a damaged file — please verify")
instead of silently trusting a partial parse. Only genuinely unrecoverable
input (password-protected, legacy .doc, or bytes with no document in them)
raises ValueError, with an actionable message.
"""

from __future__ import annotations

import base64
import html
import io
import mimetypes
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from llm import get_llm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".md"}

# Groq's currently vision-capable model (console.groq.com/docs/vision).
# It's flagged preview/not-production-ready in Groq's own docs as of this
# writing — override with GROQ_VISION_MODEL if it gets deprecated.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

# Scanned-PDF OCR cost control: CVs are 1-2 pages, postings 1-3.
MAX_OCR_PAGES = 4
OCR_RENDER_SCALE = 2  # ~144 dpi: legible for OCR without huge uploads

OCR_PROMPT = (
    "Transcribe ALL text visible in this image verbatim, preserving line "
    "breaks and structure as best you can. This is a job posting or a CV "
    "screenshot. Do not summarize, do not add commentary or explanation — "
    "output only the transcribed text."
)

MAX_IMAGE_DIMENSION = 2200  # px, plenty for OCR legibility
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # downscale above this to stay well under Groq's 20MB limit


class UnsupportedFileType(ValueError):
    pass


@dataclass
class ExtractionResult:
    text: str
    notes: list[str] = field(default_factory=list)  # things the user should know


# --- public API ---------------------------------------------------------------


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch on the file's real type. Returns text only (see extract_text_ex
    for the repair notes)."""
    return extract_text_ex(file_bytes, filename).text


def extract_text_ex(file_bytes: bytes, filename: str) -> ExtractionResult:
    suffix = Path(filename).suffix.lower()
    kind = _sniff(file_bytes, suffix)
    notes: list[str] = []

    if kind == "unknown":
        raise UnsupportedFileType(
            f"Unsupported file type '{suffix or '(none)'}'. Supported: "
            f".pdf, .docx, .txt, .md, or an image ({', '.join(sorted(IMAGE_EXTENSIONS))})."
        )
    if kind == "legacy_doc":
        raise ValueError(
            "This is an old binary Word file (.doc). Open it in Word and use "
            "'Save As' → .docx or PDF, then upload that."
        )

    expected = {".pdf": "pdf", ".docx": "docx"}.get(suffix)
    if expected and kind != expected and kind != "text":
        notes.append(f"The file is named '{suffix}' but actually contains a {kind.upper()} — parsed it as {kind.upper()}.")

    if kind == "pdf":
        text = _extract_pdf(file_bytes, notes)
    elif kind == "docx":
        text = _extract_docx(file_bytes, notes)
    elif kind == "image":
        text = _extract_image(file_bytes, filename)
    else:  # text
        text = file_bytes.decode("utf-8", errors="ignore")
    return ExtractionResult(text=text, notes=notes)


# --- type sniffing ------------------------------------------------------------

_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")


def _sniff(data: bytes, suffix: str) -> str:
    head = data[:2048]
    if b"%PDF" in head:  # some exporters prepend junk before the header
        return "pdf"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "legacy_doc"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            if suffix == ".docx":
                return "docx"  # truncated docx — let the extractor explain
        return "docx" if suffix == ".docx" else "unknown"
    if head.startswith(_IMAGE_MAGIC[:4]) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "image"
    if head.startswith(b"BM") and suffix in IMAGE_EXTENSIONS:
        return "image"
    # No recognizable magic bytes: trust the extension.
    if suffix == ".pdf":
        return "pdf"  # will fail with a clear message in _extract_pdf
    if suffix == ".docx":
        return "docx"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "unknown"


# --- PDF ----------------------------------------------------------------------

_OBJ_RE = re.compile(rb"(?<![\d])(\d+)\s+(\d+)\s+obj\b")


def _repair_pdf(data: bytes) -> bytes | None:
    """Rebuild a missing/damaged xref table + trailer by scanning for objects.

    Recovers PDFs whose body is intact but whose tail (xref / trailer /
    %%EOF) was lost — an interrupted download or a sloppy export. Returns
    None if not even a document catalog can be found (e.g. the file was
    truncated before the page tree, or uses compressed object streams).
    """
    start = data.find(b"%PDF")
    if start < 0:
        return None
    body = data[start:]
    cut = len(body)
    for marker in (b"\nxref", b"\r\nxref", b"\rxref"):
        i = body.rfind(marker)
        if i != -1:
            cut = min(cut, i)
    body = body[:cut].rstrip() + b"\n"

    offsets = {int(m.group(1)): (m.start(), int(m.group(2))) for m in _OBJ_RE.finditer(body)}
    root = None
    for num, (off, gen) in offsets.items():
        if re.search(rb"/Type\s*/Catalog\b", body[off : off + 400]):
            root = (num, gen)
    if not root:
        return None

    size = max(offsets) + 1
    out = bytearray(body)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for n in range(1, size):
        if n in offsets:
            out += b"%010d %05d n \n" % (offsets[n][0], offsets[n][1])
        else:
            out += b"0000000000 65535 f \n"
    out += b"trailer\n<< /Size %d /Root %d %d R >>\nstartxref\n%d\n%%%%EOF\n" % (
        size, root[0], root[1], xref_pos,
    )
    return bytes(out)


def _pypdf_text(data: bytes) -> tuple[str, bool]:
    """(text, was_encrypted_and_undecryptable). Raises on unparseable data."""
    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted:
        # Many "protected" PDFs only restrict editing and open with an empty
        # user password — that is fully recoverable.
        try:
            ok = reader.decrypt("")
        except Exception:  # noqa: BLE001 — e.g. missing AES backend
            ok = 0
        if not ok:
            return "", True
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip(), False


def _pdfium_doc(data: bytes):
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(data)


def _pdfium_text(data: bytes) -> str:
    doc = _pdfium_doc(data)
    return "\n\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc))).strip()


def _ocr_pdf_pages(data: bytes) -> str:
    """Scanned PDF: render pages to images and OCR them with the vision model."""
    doc = _pdfium_doc(data)
    parts: list[str] = []
    for i in range(min(len(doc), MAX_OCR_PAGES)):
        image = doc[i].render(scale=OCR_RENDER_SCALE).to_pil().convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        parts.append(_ocr_image(buf.getvalue(), "image/jpeg"))
    return "\n\n".join(p for p in parts if p).strip()


def _extract_pdf(data: bytes, notes: list[str]) -> str:
    # 1) pypdf as-is
    try:
        text, locked = _pypdf_text(data)
        if locked:
            raise ValueError(
                "This PDF is password-protected. Remove the password (or print it to a "
                "new PDF) and upload again."
            )
        if text:
            return text
        parsed_ok = True
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 — corrupt structure; try repair/other engines
        parsed_ok = False

    # 2) damaged xref/trailer: rebuild and retry
    if not parsed_ok:
        repaired = _repair_pdf(data)
        if repaired:
            try:
                text, _ = _pypdf_text(repaired)
                if text:
                    notes.append(
                        "This PDF was damaged (incomplete file); text was recovered but may be "
                        "partial — please check it below."
                    )
                    return text
            except Exception:  # noqa: BLE001
                pass

    # 3) a second, independent engine
    try:
        text = _pdfium_text(data)
        if text:
            notes.append("Parsed with a fallback PDF engine — please check the extracted text.")
            return text
        parsed_ok = True
    except Exception:  # noqa: BLE001
        pass

    # 4) no text layer at all => scanned document: OCR it automatically.
    if parsed_ok:
        try:
            text = _ocr_pdf_pages(data)
        except ValueError as exc:
            raise ValueError(
                f"This PDF has no selectable text (it's a scan) and automatic OCR failed: {exc}"
            ) from exc
        if text:
            notes.append("Scanned PDF — text was read with OCR; please check it for errors.")
            return text

    raise ValueError(
        "Couldn't recover any text from this PDF (it looks truncated or corrupt beyond "
        "repair). Re-export/re-download it, or use 'Paste text' instead."
    )


# --- DOCX ---------------------------------------------------------------------


def _docx_xml_text(data: bytes) -> str:
    """Plain text straight from the OOXML — includes text boxes, headers and
    footers, which python-docx's paragraph/table API does not expose."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [
            n for n in zf.namelist()
            if n == "word/document.xml" or re.match(r"word/(header|footer)\d*\.xml$", n)
        ]
        chunks = []
        for name in sorted(names, key=lambda n: (n != "word/document.xml", n)):
            xml = zf.read(name).decode("utf-8", errors="ignore")
            xml = re.sub(r"</w:p>", "\n", xml)
            xml = re.sub(r"<w:(?:tab|br)\b[^>]*/>", " ", xml)
            xml = re.sub(r"<[^>]+>", "", xml)
            chunks.append(html.unescape(xml))
    lines = [ln.strip() for ln in "\n".join(chunks).splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _extract_docx(data: bytes, notes: list[str]) -> str:
    primary = ""
    try:
        doc = DocxDocument(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Word docs often put content in tables (e.g. skills grids) — grab those too.
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))
        primary = "\n".join(paragraphs).strip()
    except Exception:  # noqa: BLE001 — damaged package parts (styles, rels, ...)
        primary = ""

    try:
        xml_text = _docx_xml_text(data)
    except Exception:  # noqa: BLE001 — not even a readable zip
        xml_text = ""

    if not primary and not xml_text:
        raise ValueError(
            "Couldn't read this .docx (the file is damaged or not a Word document). "
            "Re-save it from Word or upload a PDF instead."
        )
    # CV templates put content in text boxes that python-docx can't see; if
    # the raw XML holds clearly more words, it is the more complete source.
    if len(xml_text.split()) > 1.25 * len(primary.split()):
        if primary:
            notes.append("Some content (e.g. text boxes/headers) was only readable from the raw document XML.")
        return xml_text
    return primary


# --- images / OCR -------------------------------------------------------------


def _downscale_if_needed(file_bytes: bytes, mime: str) -> tuple[bytes, str]:
    """Shrink oversized screenshots so they upload fast and stay under limits.

    Re-encodes to JPEG when downscaling, so the returned mime type must be
    used as-is (don't fall back to the original filename's extension).
    """
    if len(file_bytes) <= MAX_IMAGE_BYTES:
        return file_bytes, mime

    from PIL import Image

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _ocr_image(file_bytes: bytes, mime: str) -> str:
    """Vision-model OCR. Every failure mode (bad image, no API key, rate
    limit, model retired, empty answer) surfaces as a ValueError with an
    actionable message, so callers never leak provider exceptions."""
    try:
        file_bytes, mime = _downscale_if_needed(file_bytes, mime)
    except Exception as exc:  # noqa: BLE001 — unreadable/corrupt image
        raise ValueError(f"Couldn't read this image — the file looks corrupt ({type(exc).__name__}).") from exc

    b64 = base64.b64encode(file_bytes).decode("utf-8")
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }
    try:
        llm = get_llm(temperature=0, model=VISION_MODEL)
        response = llm.invoke([message])
    except Exception as exc:  # noqa: BLE001 — no key, rate limit, model gone, network
        raise ValueError(
            f"the vision model call failed ({type(exc).__name__}: {exc}). "
            "Try again in a moment, or paste the text instead."
        ) from exc
    text = (response.content or "").strip()
    if not text:
        raise ValueError("the vision model returned no text for this image.")
    return text


def _extract_image(file_bytes: bytes, filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return _ocr_image(file_bytes, mime or "image/png")
