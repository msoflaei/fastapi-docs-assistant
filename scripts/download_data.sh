#!/usr/bin/env bash
set -e
TARGET=data/raw/fastapi
rm -rf "$TARGET"
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git "$TARGET"
cd "$TARGET"
git sparse-checkout set docs/en/docs docs_src
echo "Done. Docs are in $TARGET/docs/en/docs"