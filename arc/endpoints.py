"""Environment-driven model/endpoint configuration (ARC_*).

The single place to point the whole stack — gateway service, builder,
promotion gate, curator cron — at a different model name or a different
host:port. Every component reads these as *defaults*; explicit CLI flags
always win, so ad-hoc runs are unaffected.

  ARC_CHAT_URL     chat/completions base URL   (default http://127.0.0.1:8080/v1)
  ARC_EMBED_URL    embedding server base URL   (default http://127.0.0.1:8081)
  ARC_MODEL        model name for runtime LLM fallback (default "")
  ARC_BUILD_MODEL  model name for cache builds/promotion (default: ARC_MODEL)

On an installed server these live in /etc/arc/gateway.env (installed by
deploy/install.sh), which the systemd unit and the nightly cron both load.
"""
from __future__ import annotations

import os

DEFAULT_CHAT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_EMBED_URL = "http://127.0.0.1:8081"
# Final fallback only (keeps manual CLI runs working with no config at all).
LEGACY_DEFAULT_BUILD_MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M"


def chat_url() -> str:
    return os.environ.get("ARC_CHAT_URL", DEFAULT_CHAT_URL)


def embed_url() -> str:
    return os.environ.get("ARC_EMBED_URL", DEFAULT_EMBED_URL)


def model() -> str:
    return os.environ.get("ARC_MODEL", "")


def build_model() -> str:
    """Model used by builder/promotion; falls back to ARC_MODEL, then to the
    legacy built-in default (only relevant for unconfigured manual runs)."""
    return (os.environ.get("ARC_BUILD_MODEL")
            or os.environ.get("ARC_MODEL")
            or LEGACY_DEFAULT_BUILD_MODEL)


def family_candidate_min() -> int:
    """Miss floor for flagging a new-family seed candidate region (nightly
    curator report). Repeated misses outside every family's band that total
    at least this many over the mining window get flagged for human review —
    families are never auto-created (ARC_FAMILY_CANDIDATE_MIN, default 5)."""
    return int(os.environ.get("ARC_FAMILY_CANDIDATE_MIN", "5"))
