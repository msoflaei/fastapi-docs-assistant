"""Split cleaned documents into chunks for retrieval.

Strategy (structure-aware chunking):
1. Split each document into sections at its headings (#, ##, ###...).
2. If a section is longer than MAX_WORDS, split it further at blank lines,
   never inside a code block.
3. Start every chunk with a breadcrumb (e.g. "First Steps > Check it") and
   give it a link to its exact section on the FastAPI website.
"""
import json                 # to read and write JSON Lines files
import re                   # regular expressions, to recognize headings
from pathlib import Path    # convenient file paths
import os

IN_FILE = Path("data/processed/docs.jsonl")      # output of the cleaning step
OUT_FILE = Path("data/processed/chunks.jsonl")   # where chunks will be written
MAX_WORDS = 350   # target maximum chunk size (about 450 tokens; 1 word is roughly 1.3 tokens)
MIN_WORDS = 30    # chunks smaller than this get counted as "tiny" in the report
MERGE_BELOW = int(os.environ.get("MERGE_BELOW", 100))  # a section smaller than this gets merged with the next 

# Matches a heading line like "### Check it { #check-it }"
#   ^(#{1,6})      CAPTURE 1 to 6 "#" characters (the heading level)
#   \s+(.*?)       a space, then CAPTURE the heading text
#   \s*(?:\{\s*#([\w-]+)\s*\})?   optionally "{ #anchor }", CAPTURING the anchor name
#   \s*$           trailing spaces to the end of the line
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w-]+)\s*\})?\s*$")


def is_fence(line: str) -> bool:
    """True if the line opens or closes a ``` code block."""
    return line.strip().startswith("```")


def split_sections(text: str) -> list[dict]:
    """Split a document into sections, one per heading."""
    sections = []       # finished sections collect here
    stack = []          # the chain of headings above the current point, e.g. [(1, "First Steps"), (3, "OpenAPI")]
    # The section we're currently filling. Text before the first heading goes here.
    current = {"level": 0, "title": "", "breadcrumb": "", "anchor": None, "lines": []}
    in_code = False     # inside a code block? (Python comments start with "#" too,
                        # so we must not mistake them for headings)

    for line in text.splitlines():
        if is_fence(line):
            in_code = not in_code
        match = None if in_code else HEADING_RE.match(line)   # only look for headings outside code

        if match:                                   # a new heading starts a new section
            sections.append(current)                # save the previous section
            level = len(match.group(1))             # "###" -> 3
            title = match.group(2)                  # "OpenAPI"
            anchor = match.group(3)                 # "openapi", or None if there's no anchor
            # Keep only headings above this level, then add this one.
            # E.g. a new "###" replaces the previous "###" but keeps the "#" page title.
            stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
            current = {
                "level": level,
                "title": title,
                "breadcrumb": " > ".join(t for _, t in stack),   # "First Steps > OpenAPI"
                "anchor": anchor,
                "lines": [],
            }
        else:
            current["lines"].append(line)           # ordinary line: add it to the current section

    sections.append(current)                        # don't forget the last section
    # Drop sections with no real content (e.g. a heading immediately followed by another heading)
    return [s for s in sections if "\n".join(s["lines"]).strip()]

def section_words(section: dict) -> int:
    """Count the words in a section's content."""
    return len("\n".join(section["lines"]).split())


def absorb(target: dict, section: dict) -> None:
    """Append a section's heading and content to the end of another section."""
    heading = "#" * section["level"] + " " + section["title"]    # e.g. "### Check it" (no anchor)
    target["lines"] += ["", heading, ""] + section["lines"]


def merge_small_sections(sections: list[dict]) -> list[dict]:
    """Merge each small section into the following one, staying under MAX_WORDS."""
    merged = []
    for section in sections:
        if merged:
            prev = merged[-1]
            prev_words = section_words(prev)
            # If the previous section is small and there's room, absorb this one into it.
            if prev_words < MERGE_BELOW and prev_words + section_words(section) <= MAX_WORDS:
                absorb(prev, section)
                continue
        merged.append(section)          # otherwise this section starts a new group

    # A small LAST section has nothing after it, so merge it into the one before instead.
    if len(merged) > 1:
        last, prev = merged[-1], merged[-2]
        if section_words(last) < MERGE_BELOW and section_words(prev) + section_words(last) <= MAX_WORDS:
            absorb(prev, merged.pop())
    return merged

def split_blocks(lines: list[str]) -> list[str]:
    """Group lines into blocks (paragraphs or whole code blocks) separated by blank lines."""
    blocks, buf, in_code = [], [], False
    for line in lines:
        if is_fence(line):
            in_code = not in_code
        if line.strip() == "" and not in_code:      # a blank line outside code ends a block
            if buf:
                blocks.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)                        # blank lines INSIDE code are kept as part of the block
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def pack_blocks(blocks: list[str]) -> list[str]:
    """Combine blocks into pieces of at most MAX_WORDS, without ever splitting a block."""
    pieces, buf, count = [], [], 0
    for block in blocks:
        words = len(block.split())
        # If adding this block would make the piece too long, close the piece first.
        if buf and count + words > MAX_WORDS:
            pieces.append("\n\n".join(buf))
            buf, count = [], 0
        buf.append(block)       # a single block longer than MAX_WORDS becomes its own piece
        count += words
    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def chunk_document(doc: dict) -> list[dict]:
    """Turn one cleaned document into a list of chunk records."""
    chunks = []
    for section in merge_small_sections(split_sections(doc["text"])):
        # Link to the exact section, except for the page's top section (level 0 or 1),
        # which simply uses the page URL.
        url = doc["url"]
        if section["anchor"] and section["level"] > 1:
            url = f"{doc['url']}#{section['anchor']}"
        breadcrumb = section["breadcrumb"] or doc["title"]

        for piece in pack_blocks(split_blocks(section["lines"])):
            text = f"{breadcrumb}\n\n{piece}"       # breadcrumb first, then the content
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{len(chunks)}",   # unique ID, e.g. "tutorial/first-steps#3"
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "section": breadcrumb,
                "url": url,
                "text": text,
                "n_words": len(text.split()),
            })
    return chunks


def main() -> None:
    all_chunks = []
    with IN_FILE.open(encoding="utf-8") as f:
        for line in f:                               # each line is one document
            all_chunks.extend(chunk_document(json.loads(line)))

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Report size statistics, so we can judge whether our settings are sensible.
    sizes = sorted(c["n_words"] for c in all_chunks)
    print(f"Chunks written: {len(all_chunks)} -> {OUT_FILE}")
    print(f"Words per chunk: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"Tiny chunks (< {MIN_WORDS} words): {sum(s < MIN_WORDS for s in sizes)}")
    print(f"Oversized chunks (> {MAX_WORDS} words): {sum(s > MAX_WORDS for s in sizes)}")

    # Show a few chunks from one page so we can check them by eye.
    print("\n=== Sample chunks from tutorial/first-steps ===")
    for c in [c for c in all_chunks if c["doc_id"] == "tutorial/first-steps"][:3]:
        print(f"\n--- {c['chunk_id']} | {c['n_words']} words | {c['url']}")
        print(c["text"][:400])


if __name__ == "__main__":
    main()