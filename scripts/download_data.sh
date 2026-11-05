#!/usr/bin/env bash
# Download the FastAPI docs (and the code examples they include) at a pinned commit,
# so the data, and therefore the evaluation labels, are exactly reproducible.
set -e

# The FastAPI commit our data and evaluation labels are based on.
# Can be overridden, e.g.: FASTAPI_COMMIT=abc123 bash scripts/download_data.sh
FASTAPI_COMMIT="${FASTAPI_COMMIT:-5cb6272ed7e3e07eaf6978e1e65b94aca76fa8e0}"
TARGET=data/raw/fastapi

rm -rf "$TARGET"
mkdir -p "$TARGET"
cd "$TARGET"
git init -q
git remote add origin https://github.com/fastapi/fastapi.git
git sparse-checkout set docs/en/docs docs_src         # only these two folders
git fetch -q --depth 1 --filter=blob:none origin "$FASTAPI_COMMIT"   # just that one commit
git checkout -q FETCH_HEAD
echo "Done. FastAPI docs at commit $FASTAPI_COMMIT are in $TARGET"