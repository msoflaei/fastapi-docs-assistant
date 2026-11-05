"""Unit tests for the cleaning step."""
from pathlib import Path

from docs_assistant.clean import BASE_URL, clean_markdown, page_url


def test_admonition_becomes_plain_label(tmp_path):
    # tmp_path is a pytest "fixture": a fresh, empty temporary folder for this test.
    text = "/// tip\n\nUse the CLI.\n\n///"
    assert clean_markdown(text, tmp_path / "page.md") == "Tip:\n\nUse the CLI."


def test_admonition_with_title_keeps_title(tmp_path):
    text = "/// note | Technical Details\n\nSome text.\n\n///"
    assert clean_markdown(text, tmp_path / "page.md") == "Note: Technical Details\n\nSome text."


def test_links_keep_only_their_text(tmp_path):
    text = "See the [Tutorial](../tutorial/index.md) first."
    assert clean_markdown(text, tmp_path / "page.md") == "See the Tutorial first."


def test_html_tags_are_removed_from_prose(tmp_path):
    text = 'A <abbr title="Command Line Interface">CLI</abbr> tool.'
    assert clean_markdown(text, tmp_path / "page.md") == "A CLI tool."


def test_python_code_is_never_modified(tmp_path):
    # Looks like a link and HTML, but it's inside a Python code block, so it must stay as-is.
    text = "```python\nx = [a](b)  # <b>keep</b>\n```"
    assert clean_markdown(text, tmp_path / "page.md") == text


def test_console_blocks_lose_html_coloring(tmp_path):
    text = '```console\n$ <font color="red">fastapi</font> dev &apos;x&apos;\n```'
    assert clean_markdown(text, tmp_path / "page.md") == "```console\n$ fastapi dev 'x'\n```"


def test_code_include_is_replaced_with_file_contents(tmp_path):
    (tmp_path / "example.py").write_text("print('hi')\n")      # a fake code file to include
    text = "{* example.py hl[1] *}"
    result = clean_markdown(text, tmp_path / "page.md")
    assert result == "```python\nprint('hi')\n```"


def test_missing_include_is_dropped(tmp_path):
    text = "Before\n\n{* does_not_exist.py *}\n\nAfter"
    assert clean_markdown(text, tmp_path / "page.md") == "Before\n\nAfter"


def test_page_urls():
    assert page_url(Path("tutorial/first-steps.md")) == BASE_URL + "tutorial/first-steps/"
    assert page_url(Path("tutorial/security/index.md")) == BASE_URL + "tutorial/security/"
    assert page_url(Path("index.md")) == BASE_URL