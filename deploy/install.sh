#!/usr/bin/env bash
# CPU-ARC server installer.
#
# Installs:
#   1. a systemd service (arc-gateway) running the cache-first gateway
#   2. a cron job (nightly curator: mine production misses -> gated auto-promote)
#   3. config + state directories (/etc/arc, /var/lib/arc, /var/log/arc)
#
# Usage:
#   ./deploy/install.sh [options]
#     --repo DIR        repo checkout (default: parent of this script's dir)
#     --user USER       service user (default: arc; created if missing)
#     --python BIN      python3 binary (default: auto)
#     --schedule "M H"  cron minute hour (default: "15 3" = 03:15 daily)
#     --no-start        install but do not start the service
#     --dry-run         print actions only, change nothing
#     --uninstall       remove service + cron (+ config with --purge)
#     --purge           with --uninstall: also remove /var/lib/arc, /var/log/arc
#
# Idempotent: existing /etc/arc config files are NEVER overwritten.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SVC_USER="arc"
PYTHON="$(command -v python3 || true)"
SCHEDULE_M="15"; SCHEDULE_H="3"
START=1
DRY_RUN=0
UNINSTALL=0
PURGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$(cd "$2" && pwd)"; shift 2 ;;
    --user) SVC_USER="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --schedule) SCHEDULE_M="${2%% *}"; SCHEDULE_H="${2##* }"; shift 2 ;;
    --no-start) START=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --purge) PURGE=1; shift ;;
    *) echo "unknown option: $1 (see --help in source)" >&2; exit 2 ;;
  esac
done

run() { if [[ $DRY_RUN -eq 1 ]]; then echo "  [dry-run] $*"; else "$@"; fi; }
say() { echo; echo "==> $*"; }

have_systemctl() { command -v systemctl >/dev/null 2>&1 && systemctl list-units >/dev/null 2>&1; }

# ---------------------------------------------------------------- uninstall --
if [[ $UNINSTALL -eq 1 ]]; then
  say "uninstalling arc-gateway service + arc-curator cron"
  if have_systemctl; then
    run systemctl disable --now arc-gateway || true
    run rm -f /etc/systemd/system/arc-gateway.service
    run systemctl daemon-reload
  else
    run rm -f /etc/systemd/system/arc-gateway.service
  fi
  run rm -f /etc/cron.d/arc-curator
  run rm -rf /etc/arc
  if [[ $PURGE -eq 1 ]]; then
    say "purging state (cache, build queue, invocation log, reports)"
    run rm -rf /var/lib/arc /var/log/arc
  else
    echo "  keeping /var/lib/arc (cache + invocation log) — use --purge to remove"
  fi
  echo; echo "uninstall complete (dry-run: nothing changed)"
  exit 0
fi

# ------------------------------------------------------------------- install --
if [[ $DRY_RUN -eq 0 && $(id -u) -ne 0 ]]; then
  echo "error: run as root (or use --dry-run)" >&2; exit 1
fi
[[ -f "$REPO/arc/gateway.py" ]] || { echo "error: $REPO is not the cpu-arc repo" >&2; exit 1; }
[[ -n "$PYTHON" ]] || { echo "error: python3 not found" >&2; exit 1; }

say "CPU-ARC install"
echo "    repo:      $REPO"
echo "    user:      $SVC_USER"
echo "    python:    $PYTHON"
echo "    cron:      ${SCHEDULE_M}:${SCHEDULE_H} daily (curator)"
echo "    dry-run:   $DRY_RUN"

say "system user"
if ! id -u "$SVC_USER" >/dev/null 2>&1; then
  run useradd --system --home /var/lib/arc --shell /usr/sbin/nologin "$SVC_USER"
fi

say "state directories"
run install -d -m 0750 -o root -g "$SVC_USER" /etc/arc /var/lib/arc \
    /var/lib/arc/cache /var/lib/arc/build_queue /var/lib/arc/curate /var/log/arc

say "config (existing files are never overwritten)"
for pair in "arc-gateway.env:gateway.env" "golden-map.json:golden-map.json"; do
  src="$SCRIPT_DIR/${pair%%:*}"; dst="/etc/arc/${pair##*:}"
  if [[ -f "$dst" ]]; then
    echo "    $dst exists — leaving untouched (diff against $src if you updated templates)"
  elif [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] would install $dst (REVIEW: ports, model, thresholds)"
  else
    run install -m 0640 -o root -g "$SVC_USER" "$src" "$dst"
    echo "    installed $dst — REVIEW IT BEFORE FIRST START (ports, model, thresholds)"
  fi
done

say "systemd service (arc-gateway)"
render() { sed -e "s|__REPO__|$REPO|g" -e "s|__PYTHON__|$PYTHON|g" \
               -e "s|__USER__|$SVC_USER|g" "$1"; }
if have_systemctl; then
  render "$SCRIPT_DIR/arc-gateway.service" > /tmp/arc-gateway.service.new
  if [[ -f /etc/systemd/system/arc-gateway.service ]] \
     && ! diff -q /tmp/arc-gateway.service.new /etc/systemd/system/arc-gateway.service >/dev/null; then
    echo "    unit changed — replacing:"
    diff /etc/systemd/system/arc-gateway.service /tmp/arc-gateway.service.new | head -20 || true
  fi
  run install -m 0644 -o root -g root /tmp/arc-gateway.service.new /etc/systemd/system/arc-gateway.service
  rm -f /tmp/arc-gateway.service.new
  run systemctl daemon-reload
  if [[ $START -eq 1 ]]; then
    run systemctl enable --now arc-gateway
  else
    run systemctl enable arc-gateway
  fi
else
  echo "    warning: systemd not available — unit file only"
  run install -m 0644 -o root -g root <(render "$SCRIPT_DIR/arc-gateway.service") /etc/systemd/system/arc-gateway.service
fi

say "cron job (nightly curator)"
sed -e "s|__REPO__|$REPO|g" -e "s|__USER__|root|g" \
    -e "s|__SCHEDULE__|${SCHEDULE_M} ${SCHEDULE_H} * * *|g" \
    "$SCRIPT_DIR/arc-curator.cron" > /tmp/arc-curator.cron.new
run install -m 0644 -o root -g root /tmp/arc-curator.cron.new /etc/cron.d/arc-curator
rm -f /tmp/arc-curator.cron.new
echo "    /etc/cron.d/arc-curator: nightly mine -> gated auto-promote"

say "ownership"
# State is owned by the service user. The repo is deliberately NOT
# chowned: the gateway only reads code from it (world-readable is enough),
# and all writes go to /var/lib/arc. If you deploy the repo under the
# service user on a fresh host, chown it yourself.
run chown -R "$SVC_USER":"$SVC_USER" /var/lib/arc

echo
if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry-run complete — nothing changed"
else
  say "status"
  have_systemctl && { systemctl is-active arc-gateway || true; } || true
  test -f /etc/cron.d/arc-curator && echo "    cron installed: /etc/cron.d/arc-curator"
  cat /etc/cron.d/arc-curator 2>/dev/null | tail -1 | sed 's/^/    /' || true
  echo
  echo "next steps:"
  echo "  1. review /etc/arc/gateway.env (ARC_CHAT_URL/ARC_EMBED_URL/ARC_MODEL,"
  echo "     ports, thresholds)"
  echo "  2. build a cache into \$ARC_CACHE_DIR:"
  echo "     ARC_MERGED_OUT=\$ARC_CACHE_DIR python3 $REPO/tests/build_merged_cache.py"
  echo "     python3 -m arc.compiler \$ARC_CACHE_DIR   (from $REPO)"
  echo "  3. systemctl restart arc-gateway"
  echo "  4. first curator run: $REPO/deploy/nightly_curate.sh  (or wait for cron)"
fi
