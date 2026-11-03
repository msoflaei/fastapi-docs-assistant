"""Clean the FastAPI docs into a JSONL file ready for chunking."""
# The text in triple quotes above is a "docstring": a description of the
# module. It doesn't run; it's documentation for humans.

import json                 # to write each document as a line of JSON
import re                   # Python's regular expression (pattern matching) library
from pathlib import Path    # Path objects make working with file paths easy and safe

# ---------------------------------------------------------------------------
# Settings: where to read from and where to write to
# ---------------------------------------------------------------------------

RAW_ROOT = Path("data/raw/fastapi")        # the folder our download script created
DOCS_EN = RAW_ROOT / "docs" / "en"         # "/" joins paths: data/raw/fastapi/docs/en
DOCS_DIR = DOCS_EN / "docs"                # the folder that holds the .md files
OUT_FILE = Path("data/processed/docs.jsonl")   # where the cleaned output will be saved
BASE_URL = "https://fastapi.tiangolo.com/"     # the live website, used to build page links

# Names in CAPITAL letters are a Python convention for constants:
# values set once at the top and never changed while the program runs.

# ---------------------------------------------------------------------------
# Patterns (regular expressions) for the things we want to find and fix
# re.compile() prepares a pattern once so it can be reused quickly.
# ---------------------------------------------------------------------------

# Matches new-style code includes like: {* ../../docs_src/first_steps/tutorial001.py hl[3] *}
#   \{\*      the literal characters "{*"
#   \s*       optional spaces
#   (\S+)     CAPTURE the file path (a run of non-space characters)
#   [^*]*     then anything that isn't "*" (extra options like hl[3])
#   \*\}      the literal closing "*}"
INCLUDE_RE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")

# Matches old-style includes like: {!> ../../docs_src/x.py !}
#   \{!       the literal "{!"
#   >?        an optional ">"
#   \s*       optional spaces
#   (\S+?)    CAPTURE the path (the "?" makes it stop as early as possible)
#   \s*!\}    optional spaces, then the literal "!}"
OLD_INCLUDE_RE = re.compile(r"\{!>?\s*(\S+?)\s*!\}")

# Matches the opening line of an admonition box, like "/// tip" or "/// note | Technical Details"
#   ^         start of the line
#   /{3,}     three or more slashes (FastAPI uses 3, and 4 for nested boxes)
#   \s*(\w+)  optional spaces, then CAPTURE the kind of box (tip, note, warning...)
#   \s*(?:\|\s*(.*))?   optionally: a "|" followed by a title, which we CAPTURE
#   $         end of the line
ADMONITION_START_RE = re.compile(r"^/{3,}\s*(\w+)\s*(?:\|\s*(.*))?$")

# Matches the closing line of an admonition box: a line with only slashes, like "///"
ADMONITION_END_RE = re.compile(r"^/{3,}\s*$")

# Matches Markdown links like [Tutorial](../tutorial/index.md), and images like ![alt](pic.png)
#   !?            an optional "!" (images start with it)
#   \[([^\]]*)\]  "[", CAPTURE everything up to "]", then "]"  -> the link text
#   \([^)]*\)     "(", anything up to ")", then ")"            -> the address (discarded)
LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")

# Matches HTML tags like <abbr title="...">, </abbr>, <br>
#   </?           "<", optionally followed by "/" (closing tags)
#   [a-zA-Z]      a letter (so we don't remove things like "< 5")
#   [^>]*>        anything up to and including ">"
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Matches the page title: the first-level heading, like "# First Steps { #first-steps }"
#   ^#\s+         a line starting with a single "#" and a space
#   (.*?)         CAPTURE the title text (as little as possible)
#   \s*(?:\{\s*#[\w-]+\s*\})?   optionally the anchor "{ #first-steps }", not captured
#   \s*$          trailing spaces, end of line
# re.MULTILINE makes ^ and $ mean start/end of each LINE, not of the whole text.
H1_RE = re.compile(r"^#\s+(.*?)\s*(?:\{\s*#[\w-]+\s*\})?\s*$", re.MULTILINE)

# A dictionary that counts how many includes we found and how many failed.
# We print it at the end so we can check the cleaning worked.
stats = {"includes_resolved": 0, "includes_missing": 0}


# ---------------------------------------------------------------------------
# Function 1: replace an include pointer with the real code it points to
# ---------------------------------------------------------------------------

def resolve_include(rel_path: str, md_file: Path, fenced: bool) -> str:
    # rel_path: the path written inside the include, e.g. "../../docs_src/x.py"
    # md_file:  the .md file the include came from
    # fenced:   whether to wrap the code in ``` marks (True for new-style includes)
    # "-> str" means the function returns a string.
    """Return the code an include directive points to, or "" if not found."""

    # The include paths are relative, so we don't know for sure what they're
    # relative to. We try two likely starting folders, in order.
    for base in (DOCS_EN, md_file.parent):
        candidate = (base / rel_path).resolve()   # build the full path; resolve() handles "../"
        if candidate.is_file():                   # does a file actually exist there?
            stats["includes_resolved"] += 1       # count a success
            code = candidate.read_text(encoding="utf-8").strip()  # read the code; strip() trims blank edges
            if not fenced:                        # old-style includes already sit inside ``` marks
                return code                       # so return the bare code
            # Choose the language label for syntax highlighting: python for .py files
            lang = "python" if candidate.suffix == ".py" else ""
            return f"```{lang}\n{code}\n```"      # wrap code in a Markdown code block
            # f"..." is an f-string: values inside {} are inserted into the text.
            # \n means "new line".

    # If we get here, neither location had the file.
    print(f"Missing include in {md_file.relative_to(DOCS_DIR)}: {rel_path}")
    stats["includes_missing"] += 1                # count a failure
    return ""                                     # replace the pointer with nothing


# ---------------------------------------------------------------------------
# Function 2: clean the text of one Markdown document
# ---------------------------------------------------------------------------

def clean_markdown(text: str, md_file: Path) -> str:
    # Step A: replace every include pointer with real code.
    # .sub(function, text) finds every match and replaces it with what the
    # function returns. "lambda m: ..." is a tiny unnamed function; m is the
    # match, and m.group(1) is the first thing we CAPTURED (the file path).
    text = INCLUDE_RE.sub(lambda m: resolve_include(m.group(1), md_file, fenced=True), text)
    text = OLD_INCLUDE_RE.sub(lambda m: resolve_include(m.group(1), md_file, fenced=False), text)

    # Step B: go through the document line by line.
    out = []            # the cleaned lines collect here
    in_code = False     # are we currently inside a ``` code block?

    for line in text.splitlines():      # splitlines() turns the text into a list of lines
        stripped = line.strip()         # the line without leading/trailing spaces, for checks

        if stripped.startswith("```"):  # a ``` line opens or closes a code block
            in_code = not in_code       # flip True <-> False
            out.append(line)            # keep the ``` line itself
            continue                    # go to the next line

        if in_code:                     # inside code, we change nothing, since
            out.append(line)            # "cleaning" could break the code
            continue

        # Is this line the start of an admonition box (e.g. "/// tip")?
        start = ADMONITION_START_RE.match(stripped)   # returns a match, or None if no match
        if start:
            kind = start.group(1).capitalize()        # "tip" -> "Tip"
            title = start.group(2)                    # the optional title, or None
            # Write "Tip: Some title" if there's a title, otherwise just "Tip:"
            out.append(f"{kind}: {title}" if title else f"{kind}:")
            continue

        if ADMONITION_END_RE.match(stripped):         # a closing "///" line
            continue                                  # drop it entirely

        line = LINK_RE.sub(r"\1", line)               # replace each link with its text; \1 = first CAPTURE
        line = HTML_TAG_RE.sub("", line)              # delete HTML tags
        out.append(line)                              # keep the cleaned line

    cleaned = "\n".join(out)                          # glue the lines back into one text
    # Replace 3+ newlines in a row with exactly 2 (one blank line), then trim the ends.
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ---------------------------------------------------------------------------
# Function 3: work out the live website address for a document
# ---------------------------------------------------------------------------

def page_url(rel: Path) -> str:
    # rel is the file's path inside the docs folder, e.g. tutorial/first-steps.md
    parts = list(rel.with_suffix("").parts)   # drop ".md" and split: ["tutorial", "first-steps"]
    if parts and parts[-1] == "index":        # index.md is a folder's main page,
        parts = parts[:-1]                    # whose URL is just the folder: drop "index"
    # Join with "/" and add a trailing "/", e.g. https://fastapi.tiangolo.com/tutorial/first-steps/
    # (The homepage, with no parts, gets just the base URL.)
    return BASE_URL + "/".join(parts) + ("/" if parts else "")


# ---------------------------------------------------------------------------
# Main: run everything and write the output file
# ---------------------------------------------------------------------------

def main() -> None:                           # "-> None" means it returns nothing
    files = sorted(DOCS_DIR.rglob("*.md"))    # every .md file in all subfolders, alphabetically
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)   # create data/processed if needed
    written = 0                               # counter for documents saved

    # Open the output file for writing ("w" replaces any old version).
    # "with" closes the file automatically when the block ends.
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for md_file in files:                                  # process each document
            rel = md_file.relative_to(DOCS_DIR)                # e.g. tutorial/first-steps.md
            raw_text = md_file.read_text(encoding="utf-8")     # read the original text
            text = clean_markdown(raw_text, md_file)           # clean it
            if not text:                                       # skip empty documents
                continue

            title_match = H1_RE.search(text)                   # find the "# Title" line
            record = {                                         # one document as a dictionary
                "doc_id": rel.with_suffix("").as_posix(),      # "tutorial/first-steps", a unique ID
                "path": rel.as_posix(),                        # "tutorial/first-steps.md"
                "url": page_url(rel),                          # link to the live page
                # The title from the heading, or the file name if there's no heading
                "title": title_match.group(1) if title_match else rel.stem,
                "text": text,                                  # the cleaned content
            }
            # Convert the dictionary to JSON text and write it as one line.
            # ensure_ascii=False keeps non-English characters readable.
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    # Report what happened, so we can spot problems.
    print(f"Documents written: {written} -> {OUT_FILE}")
    print(f"Code includes resolved: {stats['includes_resolved']}")
    print(f"Code includes missing:  {stats['includes_missing']}")


# This line means: run main() only when this file is run directly
# (python -m docs_assistant.clean), not when another file imports it.
# That lets us reuse these functions later, for example in tests.
if __name__ == "__main__":
    main()