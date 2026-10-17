"""Generate FAKE results that mimic the expected behaviour of each arm, so analyze.py
can be tested without an API key. Output goes to results_synthetic/ and must never be
mixed with real runs. Nothing here is a measurement."""
import csv, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import CSV_FIELDS, ARMS, load_cfg, load_rows

cfg = load_cfg(str(Path(__file__).parent / "config.json"))
rows = load_rows(cfg)
out = Path(__file__).parent / "results_synthetic"; out.mkdir(exist_ok=True)
rng = random.Random(1)

for rep in (1, 2, 3):
    with open(out / f"rep{rep}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader()
        for arm in ARMS:
            for r in rows:
                passage_tok = int(len(r["article"].split()) * 1.33)
                tail = 120
                warm = r["is_warmup"]
                if arm == "A":
                    cw, cr, un = 0, 0, passage_tok + tail
                elif arm in ("B", "C"):
                    cw, cr, un = (passage_tok, 0, tail) if warm else (0, passage_tok, tail)
                elif arm == "D":
                    cw, cr, un = passage_tok, 0, tail
                else:  # E
                    cw, cr, un = passage_tok + tail, 0, 0
                ttft = (900 if cr == 0 else 350) + rng.gauss(0, 60)
                pred = r["gold_label"] if rng.random() < 0.82 else rng.choice([1, 2, 3, 4])
                w.writerow({
                    "ts_utc": "synthetic", "run_id": "synthetic", "rep": rep, "arm": arm, "arm_name": ARMS[arm],
                    "model": cfg["model"], "passage_id": r["passage_id"],
                    "request_index_in_passage": r["request_index_in_passage"], "is_warmup": int(warm),
                    "question_unique_id": r["question_unique_id"], "article_sha256": r["article_sha256"],
                    "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr, "input_tokens": un,
                    "output_tokens": 1, "total_input_tokens": cw + cr + un,
                    "ttft_ms": round(ttft, 1), "latency_ms": round(ttft + 80, 1), "stop_reason": "end_turn",
                    "raw_answer": "ABCD"[pred - 1], "pred_label": pred, "gold_label": r["gold_label"],
                    "correct": int(pred == r["gold_label"]), "error": "",
                })
print(f"wrote synthetic rep1..3 to {out}")
