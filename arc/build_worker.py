"""ARC Build Worker — consumes the lazy-build queue and grows the cache.

Reads pending (query, transcript) pairs from the build queue directory,
runs the 5-phase builder on each, registers new entries/functions into the
cache DB, compiles the updated cache, and signals the gateway to hot-reload.

Run:
    python -m arc.build_worker --cache-dir ./cache \
        --build-queue-dir ./build_queue \
        --chat-url http://127.0.0.1:8080/v1 \
        --embed-url http://127.0.0.1:8081 \
        --model unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M \
        [--poll-interval 5]

The worker is idempotent: if a build fails, the job stays in the queue with
status="failed" and won't be retried (avoid infinite loops on bad inputs).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("arc.build_worker")


class BuildWorker:
    """Processes the lazy-build queue, growing the ARC cache."""

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path(config["cache_dir"])
        self.queue_dir = Path(config["build_queue_dir"])
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ARC components
        from arc import Embedder, LMBackend, Thresholds, Ledger, ARC
        from arc.builder import Builder
        from arc.compiler import Compiler

        self.embedder = Embedder(backend="server", url=config["embed_url"])
        self.lm = LMBackend(url=config["chat_url"], model=config["model"])
        self.ledger = Ledger()
        self.lm.ledger = self.ledger

        thresholds = Thresholds(
            entry=config.get("entry_threshold", 0.72),
            function=config.get("function_threshold", 0.55),
            load=config.get("load_threshold", 0.45),
        )

        # Load existing cache (so new builds can reuse existing functions)
        self.arc = ARC(
            self.embedder, self.lm, thresholds,
            db_path=str(self.cache_dir), ledger=self.ledger,
        )
        self.builder = Builder(
            self.arc, builder_lm=self.lm, judge_lm=self.lm,
            max_rewrites=config.get("max_rewrites", 3),
            roundtrips=config.get("roundtrips", 2),
        )
        self.compiler = Compiler(self.arc.registry, self.entry_store)

        log.info("BuildWorker initialized: %d existing entries, %d functions",
                 len(self.arc.entry_store.entries), len(self.arc.registry.functions))

    @property
    def entry_store(self):
        return self.arc.entry_store

    def process_queue(self) -> int:
        """Process all pending jobs. Returns count of successful builds."""
        jobs = sorted(self.queue_dir.glob("build_*.json"))
        succeeded = 0
        for job_path in jobs:
            try:
                job = json.loads(job_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Skipping unreadable job %s: %s", job_path.name, e)
                continue

            if job.get("status") != "pending":
                continue

            log.info("Processing build job %s: %s", job["id"], job["query"][:60])
            t0 = time.perf_counter()

            try:
                self._build_one(job)
                job["status"] = "done"
                job["completed_at"] = time.time()
                job["duration_s"] = round(time.perf_counter() - t0, 1)
                succeeded += 1
                log.info("Build %s completed in %.1fs", job["id"], job["duration_s"])
            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
                job["completed_at"] = time.time()
                log.error("Build %s failed: %s", job["id"], e)

            job_path.write_text(json.dumps(job, indent=2))

        # If anything was built, recompile + signal gateway
        if succeeded > 0:
            self._recompile_and_notify()

        return succeeded

    def _build_one(self, job: dict) -> None:
        """Run the 5-phase builder on a single (query, transcript) pair."""
        query = job["query"]
        transcript = job["transcript"]

        # Check if we already have a similar entry (avoid duplicates)
        existing, score = self.arc.entry_store.search(query)
        if existing is not None and score > 0.95:
            log.info("Skipping %s: near-duplicate of existing entry (score=%.3f)",
                     job["id"], score)
            return

        # Run the builder
        result = self.builder.build(query, transcript)
        if not result.ok:
            raise RuntimeError(f"build failed: {result.errors}")

        log.info("Build %s: registered entry + %d functions",
                 job["id"], len(result.registered_functions))

    def _recompile_and_notify(self) -> None:
        """Recompile the cache and signal the gateway to hot-reload."""
        # Recompile all entries
        for i, entry in enumerate(self.arc.entry_store.entries):
            try:
                result = self.compiler.compile_entry(entry["description"])
                out_dir = self.cache_dir / "compiled" / f"entry_{i:03d}"
                self.compiler.write(result, out_dir)
                log.info("Compiled entry %d: %s (fully_static=%s)",
                         i, entry["description"][:50], result.all_resolved)
            except Exception as e:
                log.warning("Compile failed for entry %d: %s", i, e)

        # Write a "reload" signal file the gateway can watch
        signal_path = self.cache_dir / ".reload_signal"
        signal_path.write_text(json.dumps({"ts": time.time(), "entries": len(self.arc.entry_store.entries)}))
        log.info("Wrote reload signal (%d entries)", len(self.arc.entry_store.entries))


def main():
    parser = argparse.ArgumentParser(description="ARC Build Worker — lazy cache growth")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--build-queue-dir", default="./build_queue")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", required=True)
    parser.add_argument("--entry-threshold", type=float, default=0.72)
    parser.add_argument("--function-threshold", type=float, default=0.55)
    parser.add_argument("--load-threshold", type=float, default=0.45)
    parser.add_argument("--max-rewrites", type=int, default=3)
    parser.add_argument("--roundtrips", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between queue polls (0 = single pass)")
    parser.add_argument("--once", action="store_true", help="Process queue once and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = {
        "cache_dir": args.cache_dir,
        "build_queue_dir": args.build_queue_dir,
        "chat_url": args.chat_url,
        "embed_url": args.embed_url,
        "model": args.model,
        "entry_threshold": args.entry_threshold,
        "function_threshold": args.function_threshold,
        "load_threshold": args.load_threshold,
        "max_rewrites": args.max_rewrites,
        "roundtrips": args.roundtrips,
    }

    worker = BuildWorker(config)

    if args.once or args.poll_interval == 0:
        n = worker.process_queue()
        log.info("Single pass complete: %d builds succeeded", n)
        return

    log.info("Build worker polling every %.1fs (Ctrl+C to stop)", args.poll_interval)
    try:
        while True:
            worker.process_queue()
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        log.info("Build worker stopped")


if __name__ == "__main__":
    main()
