#!/usr/bin/env bash
# CPU-ARC nightly curator (cron payload).
# Model names/locations + all curator policy come from /etc/arc/gateway.env
# (ARC_* variables, installed by deploy/install.sh). Everything is
# overridable via environment for ad-hoc runs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ARC_ENV_FILE:-/etc/arc/gateway.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; . "$ENV_FILE"; set +a
fi

: "${ARC_CACHE_DIR:=/var/lib/arc/cache}"
: "${ARC_BUILD_QUEUE_DIR:=/var/lib/arc/build_queue}"
: "${ARC_EMBED_URL:=http://127.0.0.1:8081}"
: "${ARC_CHAT_URL:=http://127.0.0.1:8080/v1}"
: "${ARC_FAMILY_THRESHOLDS:=gcd=0.72,divisors=0.68,lcm=0.78}"

CURATOR_DB="${ARC_CURATOR_DB:-$ARC_CACHE_DIR/invocations.db}"
CURATOR_CACHE_DIR="${ARC_CURATOR_CACHE_DIR:-$ARC_CACHE_DIR}"
CURATOR_REPORT_DIR="${ARC_CURATOR_REPORT_DIR:-/var/lib/arc/curate}"
mkdir -p "$CURATOR_REPORT_DIR"

exec python3 "$REPO/eval/nightly_curate.py" \
  --db "$CURATOR_DB" \
  --cache-dir "$CURATOR_CACHE_DIR" \
  ${ARC_BUILD_QUEUE_DIR:+--build-queue-dir "$ARC_BUILD_QUEUE_DIR"} \
  --embed-url "$ARC_EMBED_URL" \
  --embed "${ARC_CURATOR_EMBED:-server}" \
  --embed-fallback auto \
  --golden-map "${ARC_CURATOR_GOLDEN_MAP:-/etc/arc/golden-map.json}" \
  --auto-promote "${ARC_CURATOR_AUTO_PROMOTE:-1}" \
  --min-repeats "${ARC_CURATOR_MIN_REPEATS:-3}" \
  --max-promotions "${ARC_CURATOR_MAX_PROMOTIONS:-2}" \
  --cases "${ARC_CURATOR_CASES:-10}" \
  --since-hours "${ARC_CURATOR_SINCE_HOURS:-168}" \
  --report-dir "$CURATOR_REPORT_DIR" \
  --family-thresholds "$ARC_FAMILY_THRESHOLDS" \
  --chat-url "$ARC_CHAT_URL" \
  ${ARC_BUILD_MODEL:+--model "$ARC_BUILD_MODEL"}
