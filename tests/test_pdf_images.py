from pathlib import Path

import pymupdf
import pytest

from furiosa_rag.pdf_images import PdfPageRenderer, RenderPixelLimitError, find_text_highlights


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


def test_default_page_rendering_is_three_times_pdf_dimensions(tmp_path: Path) -> None:
    pdf_path = tmp_path / "three-times.pdf"
    document = pymupdf.open()
    document.new_page(width=200, height=300).insert_text((20, 30), "readable source text")
    document.save(pdf_path)
    document.close()

    pixmap = pymupdf.Pixmap(PdfPageRenderer().render_png(pdf_path, 1))

    assert (pixmap.width, pixmap.height) == (600, 900)


def test_render_pixel_limit_rejects_before_pixmap_allocation(tmp_path: Path) -> None:
    pdf_path = tmp_path / "large-page.pdf"
    document = pymupdf.open()
    document.new_page(width=2_000, height=2_000)
    document.save(pdf_path)
    document.close()

    with pytest.raises(RenderPixelLimitError, match="too large to preview"):
        PdfPageRenderer(max_pixels=10_000_000).render_png(pdf_path, 1)


def test_render_pixel_limit_allows_normal_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "normal-page.pdf"
    document = pymupdf.open()
    document.new_page(width=612, height=792)
    document.save(pdf_path)
    document.close()

    png = PdfPageRenderer(max_pixels=5_000_000).render_png(pdf_path, 1)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_three_times_rendering_keeps_highlights_in_pdf_coordinates(tmp_path: Path) -> None:
    pdf_path = tmp_path / "highlight-scale.pdf"
    document = pymupdf.open()
    document.new_page(width=200, height=300).insert_text((20, 30), "stable highlight text")
    document.save(pdf_path)
    document.close()

    located = find_text_highlights(pdf_path, 1, "stable highlight text")
    pixmap = pymupdf.Pixmap(PdfPageRenderer().render_png(pdf_path, 1))

    assert (located.page_width, located.page_height) == (200.0, 300.0)
    assert (pixmap.width, pixmap.height) == (600, 900)
    assert located.rectangles
    assert 0 < located.rectangles[0].x / located.page_width < 1
    assert 0 < located.rectangles[0].y / located.page_height < 1


def test_text_highlight_lookup_returns_pdf_space_rectangle(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    document = pymupdf.open()
    document.new_page().insert_text(
        (72, 72), "BERT is designed to pre-train deep bidirectional representations."
    )
    document.save(pdf_path)
    document.close()

    located = find_text_highlights(
        pdf_path, 1, "BERT is designed to pre-train deep bidirectional representations."
    )

    assert located.page_width > 0
    assert located.page_height > 0
    assert located.rectangles
    assert located.rectangles[0].width > 0


def test_unknown_excerpt_returns_empty_highlights(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "known text")
    document.save(pdf_path)
    document.close()

    located = find_text_highlights(pdf_path, 1, "not present on this page")

    assert located.rectangles == ()


def test_highlight_prefers_body_passage_over_excerpt_header(tmp_path: Path) -> None:
    pdf_path = tmp_path / "header-and-body.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((72, 40), "Example Research Paper")
    page.insert_text(
        (72, 300),
        "The adaptive routing module chooses the visual path only when required.",
    )
    document.save(pdf_path)
    document.close()

    located = find_text_highlights(
        pdf_path,
        1,
        "Example Research Paper. "
        "The adaptive routing module chooses the visual path only when required.",
    )

    assert located.rectangles
    assert all(rectangle.y > 250 for rectangle in located.rectangles)


def test_highlight_scores_repeated_phrase_using_surrounding_context(tmp_path: Path) -> None:
    pdf_path = tmp_path / "repeated-phrase.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((72, 180), "Repeated evidence phrase appears with unrelated setup.")
    page.insert_text(
        (72, 500),
        "The adaptive module selects one route. Repeated evidence phrase appears here.",
    )
    document.save(pdf_path)
    document.close()

    located = find_text_highlights(
        pdf_path,
        1,
        "The adaptive module selects one route. Repeated evidence phrase appears here.",
    )

    assert located.rectangles
    assert all(rectangle.y > 450 for rectangle in located.rectangles)


def test_low_confidence_common_prefix_returns_no_highlight(tmp_path: Path) -> None:
    pdf_path = tmp_path / "no-confident-match.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Introduction")
    document.save(pdf_path)
    document.close()

    located = find_text_highlights(
        pdf_path, 1, "Introduction describes evidence that does not exist anywhere."
    )

    assert located.rectangles == ()


def test_low_overlap_matching_sentence_returns_no_highlight(tmp_path: Path) -> None:
    pdf_path = tmp_path / "low-overlap.pdf"
    document = pymupdf.open()
    document.new_page().insert_text(
        (72, 200), "This method improves performance across several benchmark datasets."
    )
    document.save(pdf_path)
    document.close()
    unrelated_context = " ".join(f"unrelated-{index}" for index in range(80))

    located = find_text_highlights(
        pdf_path,
        1,
        "This method improves performance across several benchmark datasets. "
        + unrelated_context,
    )

    assert located.rectangles == ()
