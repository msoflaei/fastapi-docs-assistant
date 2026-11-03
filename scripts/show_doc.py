import json
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "tutorial/first-steps"
with open("data/processed/docs.jsonl", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if doc["doc_id"] == target:
            print(doc["url"])
            print(doc["title"])
            print("-" * 60)
            print(doc["text"][:2500])
            break
    else:
        print("Not found:", target)