#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/models"
cd "$APP_DIR"

# 1) .env 준비 (없으면 템플릿 복사)
if [[ ! -f .env ]] && [[ -f .env.template ]]; then
  cp .env.template .env
fi

# 2) .env 로드(주석/빈줄 무시, export 자동)
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

# 3) 의존성 설치 (requirements.txt 있으면)
if [[ -f requirements.txt ]]; then
  python3 -m pip install --no-cache-dir -r requirements.txt
fi

# 4) Gradio 브라우저 자동 오픈 방지(코드에 inbrowser=False가 없다면 대비용)
export GRADIO_LAUNCH_IN_BROWSER="False"

# 5) 앱 실행
exec python3 /models/main.py
