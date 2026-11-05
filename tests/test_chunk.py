"""Unit tests for the chunking step."""
from docs_assistant.chunk import (
    chunk_document,
    merge_small_sections,
    pack_blocks,
    split_blocks,
    split_sections,
)

DOC_TEXT = """# First Steps { #first-steps }

Intro text.

## Check it { #check-it }

Open your browser.

```python
# this is a Python comment, not a heading
app = FastAPI()
```
"""


def test_sections_follow_headings_and_ignore_code_comments():
    sections = split_sections(DOC_TEXT)
    assert [s["breadcrumb"] for s in sections] == ["First Steps", "First Steps > Check it"]
    assert sections[1]["anchor"] == "check-it"
    # The "#" comment inside the code block stayed inside the section instead of starting a new one.
    assert any("# this is a Python comment" in line for line in sections[1]["lines"])


def test_small_sections_are_merged_with_their_heading_kept():
    merged = merge_small_sections(split_sections(DOC_TEXT))
    assert len(merged) == 1
    assert "## Check it" in "\n".join(merged[0]["lines"])


def test_large_sections_are_not_merged():
    text = "# A\n\n" + "word " * 150 + "\n\n## B\n\n" + "word " * 150
    assert len(merge_small_sections(split_sections(text))) == 2


def test_blank_lines_inside_code_do_not_split_blocks():
    lines = ["Para one.", "", "```python", "a = 1", "", "b = 2", "```", "", "Para two."]
    blocks = split_blocks(lines)
    assert len(blocks) == 3
    assert blocks[1] == "```python\na = 1\n\nb = 2\n```"


def test_pack_blocks_never_splits_a_code_block():
    long_code = "```python\n" + "x = 1\n" * 200 + "```"     # far longer than MAX_WORDS
    pieces = pack_blocks(["word " * 300, long_code])
    assert len(pieces) == 2
    assert pieces[1] == long_code                           # kept whole


def test_chunks_get_breadcrumbs_and_section_links():
    doc = {
        "doc_id": "tutorial/first-steps",
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "title": "First Steps",
        "text": "# First Steps { #first-steps }\n\n" + "word " * 150
                + "\n\n## Check it { #check-it }\n\n" + "word " * 150,
    }
    chunks = chunk_document(doc)
    assert [c["chunk_id"] for c in chunks] == ["tutorial/first-steps#0", "tutorial/first-steps#1"]
    assert chunks[0]["url"] == doc["url"]                   # top section: plain page URL
    assert chunks[1]["url"] == doc["url"] + "#check-it"     # subsection: link to the anchor
    assert chunks[1]["text"].startswith("First Steps > Check it")
    assert all("{ #" not in c["text"] for c in chunks)      # anchors removed from the text