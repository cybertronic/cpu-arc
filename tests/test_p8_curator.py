"""P8: feedback loop — miss mining (curate_misses) + gated promotion helper.

Offline (no servers):
  1-7  mine(): clustering, ranking, suggestions, queue attachment
  8    promote.py helpers: extract_answer / make_cases / case_query
  9-12 family_candidates(): new-family seed signal (volume floor, in-band
       exclusion, report-only shape)
  13-14 reload_gateway(): loop-closure hot-reload (ok + unreachable)
Live (needs embed :8081 + gateway :8083 with divisors cache):
  9-10 mine the REAL invocation log after generating repeated real misses

Run: python3 tests/test_p8_curator.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from arc import Embedder  # noqa: E402
from arc.invocations import InvocationLog  # noqa: E402
from curate_misses import mine, attach_queue, suggest  # noqa: E402
from promote import extract_answer, make_cases, case_query  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def seed_log(tmp: Path) -> InvocationLog:
    db = tmp / "invocations.db"
    log = InvocationLog(db)
    now = time.time()

    def rec(query, phase, family, score, ts, status="miss"):
        log.record(ts=ts, query=query, status=status, phase=phase,
                   family=family, entry="entry X", score=score,
                   reason=f"phase {phase}", serve_ms=10.0)

    # repeated short-style divisors miss (3 near-duplicates)
    rec("How many divisors does 48 have?", "threshold", "divisors", 0.71, now - 3600)
    rec("How many divisors does 48 have?", "threshold", "divisors", 0.71, now - 1800)
    rec("How many divisors does 49 have?", "threshold", "divisors", 0.70, now - 900)
    # separation miss
    rec("What is the greatest common divisor of 48 and 180?",
        "separation", "gcd", 0.922, now - 300)
    # far-below-band miss (definitional)
    rec("What is a Mersenne prime?", "threshold", "divisors", 0.55, now - 60)
    log.close()
    return InvocationLog(db)


def test_offline() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="p8_cur_"))
    log = seed_log(tmp)
    emb = Embedder(backend="hash")
    # hash embedder is coarse (one-token diff on 6-token text -> cosine 0.833);
    # the 0.92 default is calibrated for the semantic server embedder (live test)
    clusters = mine(log, emb, dedup_cosine=0.80)

    # 1. cluster counts
    check("clusters formed", len(clusters) >= 3, f"got {len(clusters)}")
    rep = clusters[0]
    check("repeated divisors miss is top cluster (count>=3)",
          rep["query"].startswith("How many divisors") and rep["count"] >= 3,
          f"{rep['query']!r} count={rep['count']}")
    check("near-dup variant clustered",
          any("49 have?" in v for v in rep.get("variants", [])) or
          rep["count"] >= 3, str(rep.get("variants")))

    # 2. ranking: count desc, then recency (mersenne ts now-60 > gcd now-300)
    order = [c["query"] for c in clusters]
    check("ranked by count then recency",
          order[0].startswith("How many divisors")
          and clusters[1]["query"] == "What is a Mersenne prime?",
          str(order))

    # 3. suggestions
    by_q = {c["query"]: c for c in clusters}
    sep_c = by_q["What is the greatest common divisor of 48 and 180?"]
    check("separation suggestion mentions guard",
          "guard" in sep_c["suggestion"], sep_c["suggestion"])
    mer_c = by_q["What is a Mersenne prime?"]
    check("low-score suggestion flags new-family review",
          "new-family" in mer_c["suggestion"], mer_c["suggestion"])
    div_c = rep
    check("repeated-miss suggestion points at promote.py",
          "promote.py" in div_c["suggestion"] and "divisors" in div_c["suggestion"],
          div_c["suggestion"])

    # 4. phases/score fields
    check("cluster carries phases + min score",
          "threshold" in div_c["phases"]
          and abs(div_c["min_score"] - 0.70) < 1e-9,
          f"{div_c['phases']} {div_c['min_score']}")

    # 5. queue attachment
    qdir = tmp / "queue"
    qdir.mkdir()
    (qdir / "build_1.json").write_text(json.dumps({
        "id": "build_1", "query": "How many divisors does 48 have?",
        "transcript": "... Answer: 10", "status": "pending"}))
    (qdir / "build_2.json").write_text(json.dumps({
        "id": "build_2", "query": "unrelated pending job",
        "transcript": "x", "status": "pending"}))
    attach_queue(clusters, qdir, emb)
    check("pending job attached to matching cluster",
          div_c.get("queued_job", {}).get("file") == "build_1.json"
          and div_c["queued_job"]["has_transcript"] is True,
          str(div_c.get("queued_job")))

    # 6. since filter
    since = time.time() - 120
    recent = mine(log, emb, since_s=since)
    check("since filter works",
          [c["query"] for c in recent] == ["What is a Mersenne prime?"],
          str([c["query"] for c in recent]))

    # 7. empty log
    empty = InvocationLog(tmp / "empty.db")
    check("empty log -> no candidates", mine(empty, emb) == [], "")
    empty.close()
    log.close()

    # 8. promote helpers
    check("extract_answer dict numeric", extract_answer({"the final GCD value": 6}) == 6.0,
          str(extract_answer({"the final GCD value": 6})))
    check("extract_answer string", extract_answer("Answer: 12") == 12.0,
          str(extract_answer("Answer: 12")))
    check("extract_answer nested single", extract_answer({"wrap": {"n": 4}}) == 4.0,
          str(extract_answer({"wrap": {"n": 4}})))
    import random as _r
    cases = make_cases("What is the greatest common divisor of 315 and 108?",
                       4, _r.Random(1))
    check("make_cases count+width", len(cases) == 4 and all(len(c) == 2 for c in cases)
          and all(all(10 <= n <= 99999 for n in c) for c in cases), str(cases))
    cq = case_query("What is the greatest common divisor of 315 and 108?", [777, 888])
    check("case_query substitutes numbers in order",
          cq == "What is the greatest common divisor of 777 and 888?", cq)

    # 9-12. family_candidates(): new-family seed signal (report only)
    from curate_misses import family_candidates
    now = time.time()

    def seed_region(db: Path, n: int, qfmt, family, score) -> InvocationLog:
        lg = InvocationLog(db)
        for i in range(n):
            lg.record(ts=now - (10000 - i * 600), query=qfmt(i),
                      status="miss", phase="threshold", family=family,
                      entry=None, score=score, reason="below band",
                      serve_ms=5.0)
        lg.close()
        return InvocationLog(db)

    tri = lambda i: (f"What is the area of a triangle with base {20 + i} "
                     f"and height 10?")
    c5 = mine(seed_region(tmp / "fam5.db", 5, tri, None, 0.42), emb,
              dedup_cosine=0.80)
    fc = family_candidates(c5, emb, min_misses=5, region_cosine=0.80)
    check("5 repeated out-of-band misses -> one flagged region",
          len(fc) == 1 and "triangle" in fc[0]["region_query"]
          and fc[0]["total_misses"] == 5, str(fc))
    check("region is report-only (carries a manual-seed suggestion)",
          fc and "Not auto-created" in fc[0]["suggestion"], str(fc[:1]))

    c4 = mine(seed_region(tmp / "fam4.db", 4, tri, None, 0.42), emb,
              dedup_cosine=0.80)
    check("below the volume floor (4 < 5) -> not flagged",
          family_candidates(c4, emb, min_misses=5, region_cosine=0.80) == [], "")

    div = lambda i: f"How many divisors does {40 + i} have?"
    cib = mine(seed_region(tmp / "inband.db", 6, div, "divisors", 0.71), emb,
               dedup_cosine=0.80)
    check("in-band repeated misses (family ctx, score 0.71) are not seeds",
          family_candidates(cib, emb, min_misses=5, region_cosine=0.80) == [],
          str([c["query"] for c in cib]))

    # far-below-band WITH family context (the production Mersenne shape)
    far = lambda i: (f"What is a Mersenne prime number {i} about?")
    cfar = mine(seed_region(tmp / "farband.db", 6, far, "divisors", 0.55), emb,
                dedup_cosine=0.80)
    ffc = family_candidates(cfar, emb, min_misses=5, region_cosine=0.80)
    check("below-band misses with family ctx are seeds (production shape)",
          len(ffc) == 1 and ffc[0]["total_misses"] == 6, str(ffc))

    # 13-14. reload_gateway(): loop closure — running gateway must pick up
    # promoted entries without a manual restart
    from nightly_curate import reload_gateway
    import http.server, threading as _th, socket
    hits = []

    class _ReloadHandler(http.server.BaseHTTPRequestHandler):
        # mirrors the real gateway: /reload is a POST endpoint
        def do_POST(self):
            hits.append(self.path)
            body = b'{"status": "reloaded"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ReloadHandler)
    _th.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        res = reload_gateway(f"http://127.0.0.1:{port}")
        check("reload_gateway ok on live /reload",
              res == "ok" and hits == ["/reload"], f"{res!r} {hits}")
    finally:
        srv.shutdown()
    check("reload_gateway reports failure on dead port (run must not die)",
          reload_gateway("http://127.0.0.1:1", timeout=2).startswith("FAILED"),
          "")
    return tmp


def test_live() -> None:
    # ensure the divisors gateway is up on 8083; generate real repeated misses
    port = 8083
    proc = None
    try:
        json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2))
    except Exception:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "arc.gateway", "--port", str(port),
             "--host", "127.0.0.1", "--cache-dir", "tests/gateway_cache",
             "--chat-url", "http://127.0.0.1:8080/v1",
             "--embed-url", "http://127.0.0.1:8081",
             "--model", "unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M",
             "--family-threshold", "divisors=0.725"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(6)
        try:
            json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                             timeout=2))
        except Exception:
            proc.terminate()
            check("live: gateway up on :8083", False, "gateway failed to start")
            return
    try:
        return _live_body(port)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def _live_body(port: int) -> None:

    def q(text, timeout=120):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/query",
            data=json.dumps({"query": text}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    # 9. repeated real miss (definitional; 2x) + one short-style query
    r1 = q("What is a Mersenne prime?")
    r2 = q("What is a Mersenne prime?")
    r3 = q("How many divisors does 128 have?")
    print(f"    live: mersenne x2 -> {r1['source']}/{r1.get('reason','')[:40]}, "
          f"128 -> {r3['source']}/{r3.get('reason','')[:40]}")

    log = InvocationLog("tests/gateway_cache/invocations.db")
    emb = Embedder(backend="server", url="http://127.0.0.1:8081")
    clusters = mine(log, emb)
    check("live: candidates mined from real log", len(clusters) >= 1,
          str([c["query"] for c in clusters]))
    mer = [c for c in clusters if "Mersenne" in c["query"]]
    check("live: repeated miss clustered with count>=2",
          mer and mer[0]["count"] >= 2, str(mer))
    if mer:
        print(f"    live top suggestion: {mer[0]['suggestion'][:90]}")
    log.close()


def main() -> None:
    print("[offline] curate_misses + promote helpers")
    tmp = test_offline()
    print("[live] mine real invocation log (gateway :8083 + embed :8081)")
    test_live()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nP8 CURATOR RESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
