#!/usr/bin/env bash
# scripts/sync/sync_and_smoke.sh
# Auto fetch -> rebase -> smoke test -> push. Outputs conflict hints on failure.
set -euo pipefail

REMOTE="${REMOTE:-origin}"
BASE_BRANCH="${BASE_BRANCH:-main}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "[sync] fetch..."
git fetch --prune "${REMOTE}"

echo "[sync] enable rerere (local)..."
git config rerere.enabled true
git config rerere.autoupdate true

echo "[sync] rebase onto ${REMOTE}/${BASE_BRANCH}..."
if ! git rebase "${REMOTE}/${BASE_BRANCH}"; then
  echo ""
  echo "=========================================="
  echo "[sync] REBASE CONFLICT detected."
  echo "  1. Resolve conflicts in the listed files."
  echo "  2. git add <resolved files>"
  echo "  3. git rebase --continue"
  echo "  4. Or: git rebase --abort to cancel."
  echo ""
  echo "  Tip: rerere is enabled — if you've solved"
  echo "       this conflict before, it may auto-resolve."
  echo "=========================================="
  exit 1
fi

echo "[sync] smoke tests..."
if [ -f pyproject.toml ]; then
  python3 -m pip install -U pip >/dev/null 2>&1 || true
  python3 -m pytest -q || { echo "[sync] tests failed — NOT pushing"; exit 2; }
fi

if [ -f deploy/docker/docker-compose.planA.yml ]; then
  echo "[sync] docker compose dry-run..."
  docker compose -f deploy/docker/docker-compose.planA.yml --dry-run up --build -d 2>/dev/null || true
fi

echo "[sync] push ${BRANCH}..."
git push "${REMOTE}" "${BRANCH}"

echo "[sync] done."
