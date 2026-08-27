from furiosa_rag.cli.run_multimodal_rag import _vision_exit_code


def test_require_vision_succeeds_only_when_page_and_visual_evidence_exist() -> None:
    assert _vision_exit_code(required=True, selected_page=1, used=True) == 0
    assert _vision_exit_code(required=True, selected_page=1, used=False) == 1
    assert _vision_exit_code(required=True, selected_page=None, used=False) == 1


def test_optional_vision_preserves_text_fallback_success() -> None:
    assert _vision_exit_code(required=False, selected_page=1, used=False) == 0
