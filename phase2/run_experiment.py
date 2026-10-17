"""Phase 2 harness: prefix prompt caching & explicit breakpoints on QuALITY.

Sends every row of the frozen Phase 1 selection through one or more arms and
appends one CSV row per request with the provider's usage fields.

Arms
  A  cold        no cache_control anywhere (baseline)
  B  automatic   top-level cache_control, provider picks the breakpoint
  C  explicit    cache_control on the passage block (the correct placement)
  D  sabotaged   like C, but a per-request timestamp above the passage (control: 0 hits)
  E  misplaced   cache_control on the user question block (control: writes, never reads)

Usage
  python run_experiment.py count-tokens
  python run_experiment.py smoke
  python run_experiment.py run --arms C --rep 1
  python run_experiment.py run --arms A,B,C,D,E --rep 1
  python run_experiment.py run --arms A,C --rep 1 --passages 30035,32665,99911 --config config.pilot.json
  python run_experiment.py dry-run --arms A,B,C,D,E

Requires ANTHROPIC_API_KEY in the environment (except for dry-run).
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, sys, time, datetime as dt
from pathlib import Path

ARMS = {
    "A": "cold",
    "B": "automatic",
    "C": "explicit",
    "D": "sabotaged_prefix",
    "E": "misplaced_breakpoint",
}
LETTERS = "ABCD"
CSV_FIELDS = [
    "ts_utc", "run_id", "rep", "arm", "arm_name", "model", "passage_id", "request_index_in_passage",
    "is_warmup", "question_unique_id", "article_sha256",
    "cache_creation_input_tokens", "cache_read_input_tokens", "input_tokens", "output_tokens",
    "total_input_tokens", "ttft_ms", "latency_ms", "stop_reason", "raw_answer", "pred_label",
    "gold_label", "correct", "error",
]


# ----------------------------------------------------------------------------- helpers
def load_cfg(path: str) -> dict:
    cfg = json.load(open(path))
    base = Path(path).parent
    for k in ("selection_path", "manifest_path", "results_dir"):
        cfg[k] = str((base / cfg[k]).resolve())
    return cfg


def load_rows(cfg: dict) -> list[dict]:
    return [json.loads(l) for l in open(cfg["selection_path"], encoding="utf-8")]


def question_text(row: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return f"Question: {row['question']}\n\nOptions:\n{opts}\n\nAnswer with the letter only."


ANSWER_RE = re.compile(r"^\s*(?:answer\s*:\s*)?\(?([ABCD])\)?(?=[\s.):,]|$)", re.IGNORECASE)


def parse_answer(text: str) -> int | None:
    """Accept only a bare option letter (optionally 'Answer: B', 'B.', '(B)').

    A sentence that merely starts with one of the letters, e.g. 'Count: Grampa,' or
    'Because ...', is a non-answer and must not be credited.
    """
    m = ANSWER_RE.match(text or "")
    return LETTERS.index(m.group(1).upper()) + 1 if m else None   # gold_label is 1-indexed


# ----------------------------------------------------------------------------- request builders
def build_request(arm: str, row: dict, cfg: dict, run_tag: str) -> tuple[dict, dict]:
    """Return (kwargs for messages.create/stream, extra_body).

    The run_tag (arm + rep) is embedded in the instructions block so every (arm, rep)
    owns a distinct prefix. That stops arm A from accidentally reading entries that
    arm C wrote for the same passage, and lets arms run back to back.
    """
    instructions = f"{cfg['instructions']}\n\n[experiment run: {run_tag}]"
    if arm == "D":
        # Changing content ABOVE the breakpoint. Guarantees a different prefix hash every request.
        instructions = f"[request issued at {dt.datetime.now(dt.timezone.utc).isoformat()}]\n" + instructions

    instr_block = {"type": "text", "text": instructions}
    passage_block = {"type": "text", "text": row["article"]}
    user_block = {"type": "text", "text": question_text(row)}

    if arm in ("C", "D"):
        passage_block["cache_control"] = {"type": "ephemeral"}
    if arm == "E":
        user_block["cache_control"] = {"type": "ephemeral"}

    kwargs = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        # Sonnet 5.5 rejects sampling params (temperature/top_p/top_k) and thinks by default.
        # Thinking is pinned off via config so every arm sends identical thinking settings;
        # any difference here would invalidate the cache between arms.
        "thinking": cfg["thinking"],
        "output_config": cfg["output_config"],
        "system": [instr_block, passage_block],
        "messages": [{"role": "user", "content": [user_block]}],
    }
    extra_body = {}
    if arm == "B":
        # Automatic caching: top-level cache_control, provider moves the breakpoint itself.
        extra_body["cache_control"] = {"type": "ephemeral"}
    return kwargs, extra_body


# ----------------------------------------------------------------------------- API calls
def get_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    import anthropic
    return anthropic.Anthropic(max_retries=3, timeout=120)


def send(client, kwargs: dict, extra_body: dict) -> dict:
    """Stream the request so we can measure time to first token. Returns a flat dict."""
    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    with client.messages.stream(**kwargs, extra_body=extra_body or None) as stream:
        for event in stream:
            if ttft is None and event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                ttft = time.perf_counter() - t0
                text_parts.append(event.delta.text)
            elif event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                text_parts.append(event.delta.text)
        final = stream.get_final_message()
    latency = time.perf_counter() - t0
    u = final.usage
    return {
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "ttft_ms": round((ttft or latency) * 1000, 1),
        "latency_ms": round(latency * 1000, 1),
        "stop_reason": final.stop_reason,
        "raw_answer": "".join(text_parts),
    }


# ----------------------------------------------------------------------------- commands
def cmd_count_tokens(cfg: dict) -> None:
    """Fill manifest.passages[*].provider_token_count and refuse passages under the minimum."""
    client = get_client()
    manifest = json.load(open(cfg["manifest_path"]))
    rows = load_rows(cfg)
    first_row = {r["passage_id"]: r for r in reversed(rows)}  # any row per passage
    bad = []
    for p in manifest["passages"]:
        row = first_row[p["article_id"]]
        kwargs, _ = build_request("A", row, cfg, "count")
        res = client.messages.count_tokens(model=cfg["model"], system=kwargs["system"], messages=kwargs["messages"])
        # whole-request count; the passage is ~98% of it, which is what we care about for the minimum
        p["provider_token_count"] = res.input_tokens
        p["provider_token_count_model"] = cfg["model"]
        status = "ok" if res.input_tokens >= cfg["min_cacheable_prefix_tokens"] else "BELOW MINIMUM"
        if status != "ok":
            bad.append(p["article_id"])
        print(f"{p['article_id']:>6}  est={p['est_tokens']:>5}  provider={res.input_tokens:>6}  {status}  {p['title'][:40]}")
    manifest["token_count_model"] = cfg["model"]
    manifest["token_count_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    json.dump(manifest, open(cfg["manifest_path"], "w"), indent=2)
    print(f"\nmanifest updated: {cfg['manifest_path']}")
    if bad:
        sys.exit(f"Passages below the {cfg['min_cacheable_prefix_tokens']}-token minimum: {bad}. Remove them before running.")


def cmd_dry_run(cfg: dict, arms: list[str]) -> None:
    rows = load_rows(cfg)
    row = rows[1]  # a non-warmup row
    for arm in arms:
        kwargs, extra = build_request(arm, row, cfg, f"{arm}-dry")
        # redact the passage so the printout is readable
        shown = json.loads(json.dumps(kwargs))
        shown["system"][1]["text"] = f"<passage {len(row['article'])} chars, sha256 {row['article_sha256'][:12]}...>"
        print(f"\n=== arm {arm} ({ARMS[arm]}) ===")
        print(json.dumps({"kwargs": shown, "extra_body": extra}, indent=2))


def existing_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {(r["arm"], r["rep"], r["question_unique_id"]) for r in csv.DictReader(f) if not r["error"]}


def select_passages(all_ids: list[str], rep: int, limit: int | None, wanted: list[str] | None) -> list[str]:
    """Passage order for a run: seeded shuffle per rep, then optionally restricted.

    --passages keeps only the named ids (cheap pilots: pick the shortest passages).
    --limit-passages keeps the first N of the shuffled order.
    """
    order = list(all_ids)
    random.Random(20261017 + rep).shuffle(order)
    if wanted:
        unknown = [w for w in wanted if w not in all_ids]
        if unknown:
            sys.exit(f"unknown passage ids {unknown}; known: {sorted(all_ids)}")
        order = [pid for pid in order if pid in wanted]
    if limit:
        order = order[:limit]
    return order


def cmd_run(cfg: dict, arms: list[str], rep: int, limit_passages: int | None, smoke: bool,
            passage_ids: list[str] | None = None) -> None:
    client = get_client()
    rows = load_rows(cfg)
    results_dir = Path(cfg["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = results_dir / ("smoke.csv" if smoke else f"rep{rep}.csv")
    done = existing_keys(csv_path)

    # group rows by passage, keep within-passage order, shuffle passage order per rep
    passages: dict[str, list[dict]] = {}
    for r in rows:
        passages.setdefault(str(r["passage_id"]), []).append(r)
    order = select_passages(sorted(passages), rep, limit_passages, passage_ids)
    if smoke:
        order, arms = order[:1], ["C"]
        passages[order[0]] = passages[order[0]][:3]

    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for arm in arms:
            run_tag = f"{arm}-rep{rep}" if not smoke else f"smoke-{run_id}"
            print(f"\n### arm {arm} ({ARMS[arm]}), rep {rep}, {len(order)} passages")
            for pid in order:
                for row in passages[pid]:
                    key = (arm, str(rep), row["question_unique_id"])
                    if key in done:
                        continue
                    kwargs, extra = build_request(arm, row, cfg, run_tag)
                    assert kwargs["system"][1]["text"] == row["article"], "passage block must be byte-identical to the row"
                    out = {
                        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "run_id": run_id, "rep": rep,
                        "arm": arm, "arm_name": ARMS[arm], "model": cfg["model"], "passage_id": pid,
                        "request_index_in_passage": row["request_index_in_passage"], "is_warmup": int(row["is_warmup"]),
                        "question_unique_id": row["question_unique_id"], "article_sha256": row["article_sha256"],
                        "gold_label": row["gold_label"], "error": "",
                    }
                    try:
                        res = send(client, kwargs, extra)
                        out.update(res)
                        out["total_input_tokens"] = res["cache_creation_input_tokens"] + res["cache_read_input_tokens"] + res["input_tokens"]
                        out["pred_label"] = parse_answer(res["raw_answer"])
                        out["correct"] = int(out["pred_label"] == row["gold_label"]) if out["pred_label"] else 0
                    except Exception as e:  # log and continue; resume will retry it
                        out["error"] = f"{type(e).__name__}: {e}"[:300]
                    w.writerow({k: out.get(k, "") for k in CSV_FIELDS}); f.flush()
                    if "credit balance" in out["error"] or "authentication" in out["error"].lower():
                        sys.exit(f"\nStopping: {out['error']}\n"
                                 f"Fix billing/key, then rerun the same command; it resumes from {csv_path.name}.")
                    print(f"  {pid} #{row['request_index_in_passage']} warm={row['is_warmup']!s:5} "
                          f"write={out.get('cache_creation_input_tokens','-'):>6} read={out.get('cache_read_input_tokens','-'):>6} "
                          f"uncached={out.get('input_tokens','-'):>5} ttft={out.get('ttft_ms','-')}ms "
                          f"ans={out.get('raw_answer','')!r} ok={out.get('correct','-')} {out['error']}")
                    time.sleep(cfg["inter_request_sleep_s"])
                time.sleep(cfg["inter_passage_sleep_s"])
    print(f"\nwrote {csv_path}")
    if smoke:
        print("\nExpected: request #0 write≈passage tokens, read=0; requests #1 and #2 read≈passage tokens, write=0.\n"
              "If all three show write=0 and read=0, the prefix is below the minimum or the SDK dropped cache_control.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["count-tokens", "smoke", "run", "dry-run"])
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
    ap.add_argument("--arms", default="A,B,C,D,E")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--limit-passages", type=int, help="first N passages of the shuffled order")
    ap.add_argument("--passages", help="comma-separated passage ids to run, e.g. 30035,32665,99911")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    arms = [x.strip().upper() for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            sys.exit(f"unknown arm {x}")
    if a.command == "count-tokens":
        cmd_count_tokens(cfg)
    elif a.command == "dry-run":
        cmd_dry_run(cfg, arms)
    else:
        wanted = [x.strip() for x in a.passages.split(",") if x.strip()] if a.passages else None
        if a.command == "smoke":
            cmd_run(cfg, ["C"], a.rep, None, smoke=True, passage_ids=wanted)
        else:
            cmd_run(cfg, arms, a.rep, a.limit_passages, smoke=False, passage_ids=wanted)


if __name__ == "__main__":
    main()
