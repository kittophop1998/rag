#!/usr/bin/env bash
# Convenience script to start the Company RAG service in dev mode.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "==> Creating virtualenv (.venv)"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies"
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  echo "==> Creating .env from .env.example (fill it in!)"
  cp .env.example .env
fi

echo "==> Starting FastAPI on http://localhost:${PORT:-8000}"
exec uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --reload
