import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "response_regeneration"
    / "prepare_aya.py"
)

spec = importlib.util.spec_from_file_location("prepare_aya", MODULE_PATH)
prepare_aya = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prepare_aya)


def test_row_to_conversation_converts_aya_row() -> None:
    row = {
        "inputs": "Hello there",
        "targets": "Hi back",
        "language": "English",
        "language_code": "eng",
        "annotation_type": "original-annotations",
        "user_id": "abc123",
    }

    convo = prepare_aya.row_to_conversation(row)

    assert convo is not None
    assert convo["messages"][0]["content"] == "Hello there"
    assert convo["messages"][1]["content"] == "Hi back"
    assert convo["conversations"][0]["value"] == "Hello there"
    assert convo["conversations"][1]["value"] == "Hi back"
    assert convo["source"] == "Aya"
    assert convo["language"] == "English"
    assert convo["id"].startswith("aya_abc123")
