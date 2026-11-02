from pathlib import Path

docs_dir = Path("data/raw/fastapi/docs/en/docs")
files = sorted(docs_dir.rglob("*.md"))

total_words = 0
for f in files:
    total_words += len(f.read_text(encoding="utf-8").split())

print(f"Markdown files: {len(files)}")
print(f"Total words: {total_words:,}")
print("Sample files:")
for f in files[:10]:
    print("  ", f.relative_to(docs_dir))