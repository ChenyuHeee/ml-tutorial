#!/usr/bin/env bash
set -euo pipefail

# Publish static viewer + tutorial content to gh-pages branch.
# Usage:
#   ./scripts/publish-gh-pages.sh
#   ./scripts/publish-gh-pages.sh --push
#
# Requires: git

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SITE_DIR="$ROOT_DIR/site"
TUTORIAL_DIR="$ROOT_DIR/ml-tutorial"
CURRICULUM_DIR="$ROOT_DIR/ml-curriculum"
BRANCH="gh-pages"
DO_PUSH=false

if [[ "${1:-}" == "--push" ]]; then
  DO_PUSH=true
fi

if [[ ! -d "$SITE_DIR" ]]; then
  echo "site/ not found. Aborting." >&2
  exit 1
fi

if [[ ! -d "$TUTORIAL_DIR" ]]; then
  echo "ml-tutorial/ not found. Aborting." >&2
  exit 1
fi

if [[ ! -d "$CURRICULUM_DIR" ]]; then
  echo "ml-curriculum/ not found. Aborting." >&2
  exit 1
fi

if ! git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repo. Run 'git init' first." >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Build publish payload
mkdir -p "$TMP_DIR/content"
cp -R "$SITE_DIR"/* "$TMP_DIR/"
cp -R "$TUTORIAL_DIR" "$TMP_DIR/content/ml-tutorial"
cp -R "$CURRICULUM_DIR" "$TMP_DIR/content/ml-curriculum"

# Switch branch
cd "$ROOT_DIR"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
else
  git switch --orphan "$BRANCH"
fi

# Replace contents
# shellcheck disable=SC2115
rm -rf ./*
cp -R "$TMP_DIR"/* .

# Keep Pages clean
find . -name '._*' -delete || true
find . -name '.DS_Store' -delete || true

git add -A
if git diff --cached --quiet; then
  echo "No changes to publish."
else
  git commit -m "Deploy: update GitHub Pages content"
fi

if [[ "$DO_PUSH" == true ]]; then
  git push -u origin "$BRANCH"
  echo "Pushed $BRANCH to origin."
else
  echo "Created/updated local '$BRANCH' branch. Use --push to push to origin."
fi

# Back to previous branch
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  git switch "$CURRENT_BRANCH" >/dev/null 2>&1 || true
fi
