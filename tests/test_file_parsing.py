"""Tests for local (non-network) file parsing paths. Vision/OCR extraction
isn't covered here since it requires a live Groq call."""

from __future__ import annotations

import io

import pytest
from docx import Document
from pypdf import PdfWriter

import file_parsing


class _FakeVision:
    """Stands in for the Groq vision model. Records calls."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1

        class _R:
            content = self.reply

        return _R()


def make_pdf(text: str) -> bytes:
    """Minimal single-page text PDF, built by hand (no reportlab needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def test_extract_docx_including_tables():
    doc = Document()
    doc.add_paragraph("Ada Yilmaz")
    doc.add_paragraph("Backend Software Engineer")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Python"
    table.rows[0].cells[1].text = "FastAPI"
    buf = io.BytesIO()
    doc.save(buf)

    text = file_parsing.extract_text(buf.getvalue(), "cv.docx")

    assert "Ada Yilmaz" in text
    assert "Python | FastAPI" in text


def test_extract_pdf_with_no_text_layer_raises(monkeypatch):
    # No text layer => OCR fallback. Stub the vision model so this never makes a live call.
    monkeypatch.setattr(file_parsing, "get_llm", lambda **kw: _FakeVision(""))
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(ValueError):
        file_parsing.extract_text(buf.getvalue(), "blank.pdf")


def test_extract_plain_text_file():
    text = file_parsing.extract_text(b"hello world", "notes.txt")
    assert text == "hello world"


def test_unsupported_extension_raises():
    with pytest.raises(file_parsing.UnsupportedFileType):
        file_parsing.extract_text(b"whatever", "resume.exe")


def test_downscale_if_needed_shrinks_large_images(monkeypatch):
    from PIL import Image

    img = Image.new("RGB", (3000, 3000), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    monkeypatch.setattr(file_parsing, "MAX_IMAGE_BYTES", 0)  # force the downscale branch
    shrunk_bytes, mime = file_parsing._downscale_if_needed(buf.getvalue(), "image/png")

    assert mime == "image/jpeg"
    shrunk = Image.open(io.BytesIO(shrunk_bytes))
    assert max(shrunk.size) <= file_parsing.MAX_IMAGE_DIMENSION


def test_downscale_if_needed_leaves_small_images_alone():
    from PIL import Image

    img = Image.new("RGB", (50, 50), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    original_bytes = buf.getvalue()

    result_bytes, mime = file_parsing._downscale_if_needed(original_bytes, "image/png")

    assert result_bytes == original_bytes
    assert mime == "image/png"


# --- real recovery paths (not just nicer error messages) --------------------

TEXT = "Python LangGraph FastAPI engineer"


def test_pdf_text_baseline():
    assert TEXT in file_parsing.extract_text(make_pdf(TEXT), "cv.pdf")


def test_pdf_with_junk_before_header_is_recovered():
    text = file_parsing.extract_text(b"\n\nGARBAGE" + make_pdf(TEXT), "cv.pdf")
    assert TEXT in text


@pytest.mark.parametrize("cut", [40, 90, 150])
def test_pdf_with_lost_trailer_is_repaired_and_flagged(cut):
    damaged = make_pdf(TEXT)[:-cut]
    result = file_parsing.extract_text_ex(damaged, "cv.pdf")
    assert TEXT in result.text
    assert any("damaged" in n for n in result.notes)


def test_pdf_with_empty_user_password_is_decrypted():
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(TEXT)))
    writer.encrypt(user_password="", owner_password="owner", algorithm="RC4-128")
    buf = io.BytesIO()
    writer.write(buf)
    assert TEXT in file_parsing.extract_text(buf.getvalue(), "cv.pdf")


def test_pdf_with_real_password_gives_actionable_error():
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(TEXT)))
    writer.encrypt(user_password="secret", owner_password="owner", algorithm="RC4-128")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ValueError, match="password"):
        file_parsing.extract_text(buf.getvalue(), "cv.pdf")


def test_scanned_pdf_is_ocrd_automatically(monkeypatch):
    fake = _FakeVision("SCANNED CV Python")
    monkeypatch.setattr(file_parsing, "get_llm", lambda **kw: fake)
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)  # no text layer
    buf = io.BytesIO()
    writer.write(buf)

    result = file_parsing.extract_text_ex(buf.getvalue(), "scan.pdf")

    assert result.text == "SCANNED CV Python"
    assert fake.calls == 1
    assert any("OCR" in n for n in result.notes)


def test_scanned_pdf_ocr_is_capped_to_max_pages(monkeypatch):
    fake = _FakeVision("page text")
    monkeypatch.setattr(file_parsing, "get_llm", lambda **kw: fake)
    writer = PdfWriter()
    for _ in range(file_parsing.MAX_OCR_PAGES + 3):
        writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    file_parsing.extract_text(buf.getvalue(), "scan.pdf")
    assert fake.calls == file_parsing.MAX_OCR_PAGES


def test_scanned_pdf_ocr_failure_is_actionable(monkeypatch):
    def boom(**kw):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(file_parsing, "get_llm", boom)
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ValueError, match="OCR failed"):
        file_parsing.extract_text(buf.getvalue(), "scan.pdf")


def test_unrecoverable_pdf_says_what_to_do():
    with pytest.raises(ValueError, match="Paste text"):
        file_parsing.extract_text(b"%PDF-1.4 truncated garbage", "cv.pdf")


def test_file_content_wins_over_extension():
    # A Word file that was renamed to .pdf still gets parsed as Word.
    doc = Document()
    doc.add_paragraph("Ada Yilmaz backend engineer")
    buf = io.BytesIO()
    doc.save(buf)
    result = file_parsing.extract_text_ex(buf.getvalue(), "cv.pdf")
    assert "Ada Yilmaz" in result.text
    assert any("actually contains a DOCX" in n for n in result.notes)


def _docx_with_textbox() -> bytes:
    import zipfile

    doc = Document()
    doc.add_paragraph("Ada")
    buf = io.BytesIO()
    doc.save(buf)
    src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for item in src.namelist():
            data = src.read(item)
            if item == "word/document.xml":
                box = (
                    "<w:p><w:r><w:pict><w:txbxContent><w:p><w:r><w:t>"
                    "SKILLS Python Docker Kubernetes Terraform PostgreSQL Redis Kafka FastAPI"
                    "</w:t></w:r></w:p></w:txbxContent></w:pict></w:r></w:p>"
                )
                data = data.decode().replace("</w:body>", box + "</w:body>").encode()
            dst.writestr(item, data)
    return out.getvalue()


def test_docx_text_boxes_are_not_lost():
    result = file_parsing.extract_text_ex(_docx_with_textbox(), "cv.docx")
    assert "Kubernetes" in result.text and "Ada" in result.text


def test_damaged_docx_package_falls_back_to_raw_xml():
    import zipfile

    doc = Document()
    doc.add_paragraph("Ada Yilmaz backend engineer with FastAPI")
    buf = io.BytesIO()
    doc.save(buf)
    src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:  # drop [Content_Types].xml: python-docx refuses it
        for item in src.namelist():
            if item != "[Content_Types].xml":
                dst.writestr(item, src.read(item))
    text = file_parsing.extract_text(out.getvalue(), "cv.docx")
    assert "FastAPI" in text


def test_truncated_docx_gives_actionable_error():
    with pytest.raises(ValueError, match="damaged"):
        file_parsing.extract_text(b"PK\x03\x04 truncated", "cv.docx")


def test_legacy_doc_gets_conversion_instructions():
    with pytest.raises(ValueError, match="Save As"):
        file_parsing.extract_text(b"\xd0\xcf\x11\xe0" + b"\x00" * 64, "cv.doc")


def test_corrupt_large_image_raises_value_error():
    with pytest.raises(ValueError, match="corrupt"):
        file_parsing.extract_text(b"\x89PNG\r\n\x1a\n" + b"x" * (5 * 1024 * 1024), "shot.png")


def test_vision_model_failure_raises_value_error(monkeypatch):
    def boom(**kw):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(file_parsing, "get_llm", boom)
    with pytest.raises(ValueError, match="vision model call failed"):
        file_parsing.extract_text(b"\x89PNG\r\n\x1a\n tiny", "shot.png")
