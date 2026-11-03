import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PATH = re.compile(r"\b((?:app|scripts|docs|tests)/[\w./-]+\.(?:py|sh|md|json|png))")

# Sentences that mention the word only to explain why it is not used.
ALLOWED = (
    'and never "verified"',
)


def test_every_file_the_readme_names_exists():
    named = sorted(set(PATH.findall(README)))
    assert named, "the README should point at real files"
    missing = [path for path in named if not (ROOT / path).exists()]
    assert missing == []


def test_the_readme_never_calls_the_checks_a_guarantee():
    assert "not mean the claim is correct" in README
    text = README.replace("verifiable", "").replace("unverified", "")
    for phrase in ALLOWED:
        assert phrase in text, f"expected phrase is gone: {phrase!r}"
        text = text.replace(phrase, "")
    assert "verified" not in text.casefold()


def test_the_images_are_small_enough_and_the_diagram_is_still_there():
    images = sorted((ROOT / "docs" / "images").glob("*.png"))
    assert images, "the README should have screenshots"
    assert [p.name for p in images if p.stat().st_size > 3_000_000] == []
    assert "flowchart" in README


def test_the_generated_module_map_is_credited():
    assert "docs/images/module-map.png" in README
    assert "GitDiagram" in README


def test_both_diagrams_are_text_diagrams_and_the_module_map_image_is_only_linked():
    assert README.count("flowchart") == 2
    assert "![Module map" not in README
    assert "(docs/images/module-map.png)" in README
