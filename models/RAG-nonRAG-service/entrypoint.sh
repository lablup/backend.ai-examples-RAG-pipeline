#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/models"
cd "$APP_DIR"

# 1) prepare .env
if [[ ! -f .env ]] && [[ -f .env.template ]]; then
  cp .env.template .env
fi

# 2) load .env
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  . ./.env
  set +a
fi

# 3) Install dependencies (if requirements.txt exists)
if [[ -f requirements.txt ]]; then
  python3 -m pip install --no-cache-dir -r requirements.txt
fi

# 4) Prevent Gradio browser from auto-opening (for cases where inbrowser=False is not in the code)
export GRADIO_LAUNCH_IN_BROWSER="False"

# 5) Run the app
exec python3 /models/main.py
