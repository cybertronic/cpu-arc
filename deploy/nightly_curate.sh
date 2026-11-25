#!/usr/bin/env bash
# CPU-ARC nightly curator (cron payload).
# Config comes from /etc/arc/gateway.env (installed by deploy/install.sh);
# everything is overridable via environment for ad-hoc runs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ARC_ENV_FILE:-/etc/arc/gateway.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; . "$ENV_FILE"; set +a
fi

: "${CACHE_DIR:=/var/lib/arc/cache}"
: "${BUILD_QUEUE_DIR:=/var/lib/arc/build_queue}"
: "${EMBED_URL:=http://127.0.0.1:8081}"
: "${CHAT_URL:=http://127.0.0.1:8080/v1}"
: "${FAMILY_THRESHOLDS:=gcd=0.72,divisors=0.68,lcm=0.78}"

CURATOR_DB="${CURATOR_DB:-$CACHE_DIR/invocations.db}"
CURATOR_CACHE_DIR="${CURATOR_CACHE_DIR:-$CACHE_DIR}"
CURATOR_REPORT_DIR="${CURATOR_REPORT_DIR:-/var/lib/arc/curate}"
mkdir -p "$CURATOR_REPORT_DIR"

exec python3 "$REPO/eval/nightly_curate.py" \
  --db "$CURATOR_DB" \
  --cache-dir "$CURATOR_CACHE_DIR" \
  ${BUILD_QUEUE_DIR:+--build-queue-dir "$BUILD_QUEUE_DIR"} \
  --embed-url "$EMBED_URL" \
  --embed "${CURATOR_EMBED:-server}" \
  --embed-fallback auto \
  --golden-map "${CURATOR_GOLDEN_MAP:-/etc/arc/golden-map.json}" \
  --auto-promote "${CURATOR_AUTO_PROMOTE:-1}" \
  --min-repeats "${CURATOR_MIN_REPEATS:-3}" \
  --max-promotions "${CURATOR_MAX_PROMOTIONS:-2}" \
  --cases "${CURATOR_CASES:-10}" \
  --since-hours "${CURATOR_SINCE_HOURS:-168}" \
  --report-dir "$CURATOR_REPORT_DIR" \
  --family-thresholds "$FAMILY_THRESHOLDS" \
  --chat-url "$CHAT_URL" \
  --model "${MODEL:-unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M}"
