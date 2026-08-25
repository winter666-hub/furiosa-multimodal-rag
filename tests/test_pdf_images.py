from pathlib import Path

import pymupdf
import pytest

from furiosa_rag.pdf_images import PdfPageRenderer


def test_pdf_renderer_renders_one_based_selected_page_to_png(tmp_path: Path) -> None:
    pdf_path = tmp_path / "pages.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "first")
    document.new_page().insert_text((72, 72), "second")
    document.save(pdf_path)
    document.close()

    renderer = PdfPageRenderer(dpi=72)
    png = renderer.render_png(pdf_path, 2)
    data_url = renderer.render_data_url(pdf_path, 2)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert data_url.startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="one-based"):
        renderer.render_png(pdf_path, 0)
