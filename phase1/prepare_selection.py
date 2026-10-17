"""Phase 1: freeze a QuALITY passage selection for the prompt-caching experiment.

Input : QuALITY v1.0.1 dev split (htmlstripped), one JSON object per line.
Output: selection.jsonl  (one row per question, frozen order)
        manifest.json    (what was selected and why)

Token counts here are an ESTIMATE (words * 1.33). Phase 2 must re-count every
passage with the provider's token-count endpoint and refuse any passage below
the model's minimum cacheable prefix.
"""
import json, random, sys, hashlib
from collections import defaultdict

SRC = sys.argv[1]
OUT_DIR = sys.argv[2]
SEED = 20261017
MIN_EST_TOKENS, MAX_EST_TOKENS = 3500, 8000
MIN_QUESTIONS, N_PASSAGES, MAX_Q_PER_PASSAGE = 5, 20, 8
WORDS_PER_TOKEN = 1 / 1.33

def est_tokens(text: str) -> int:
    return int(len(text.split()) / WORDS_PER_TOKEN)

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]

# The same article can appear in several rows (different writer sets). Merge by article_id.
articles = {}
questions = defaultdict(list)
for r in rows:
    aid = r["article_id"]
    articles.setdefault(aid, {
        "article_id": aid, "title": r["title"], "author": r["author"],
        "year": r["year"], "source": r["source"], "url": r["url"],
        "article": r["article"],
    })
    for q in r["questions"]:
        # keep only questions annotators marked answerable and that have a gold label
        if q.get("gold_label") is None:
            continue
        questions[aid].append({
            "question_unique_id": q["question_unique_id"],
            "question": q["question"],
            "options": q["options"],
            "gold_label": q["gold_label"],          # 1-indexed in QuALITY
            "difficult": q.get("difficult", 0),
        })

candidates = []
for aid, a in articles.items():
    t = est_tokens(a["article"])
    nq = len(questions[aid])
    if MIN_EST_TOKENS <= t <= MAX_EST_TOKENS and nq >= MIN_QUESTIONS:
        candidates.append((aid, t, nq))

rng = random.Random(SEED)
rng.shuffle(candidates)
chosen = sorted(candidates[:N_PASSAGES], key=lambda x: x[0])

sel_rows, manifest_passages = [], []
for aid, t, nq in chosen:
    a = articles[aid]
    qs = questions[aid][:]
    rng.shuffle(qs)
    qs = qs[:MAX_Q_PER_PASSAGE]
    sha = hashlib.sha256(a["article"].encode("utf-8")).hexdigest()
    manifest_passages.append({
        "article_id": aid, "title": a["title"], "author": a["author"], "year": a["year"],
        "est_tokens": t, "n_questions": len(qs), "article_sha256": sha,
        "provider_token_count": None,   # Phase 2 fills this from the count_tokens endpoint
    })
    for i, q in enumerate(qs):
        sel_rows.append({
            "passage_id": aid, "request_index_in_passage": i, "is_warmup": i == 0,
            "article_sha256": sha, "article": a["article"],
            **q,
        })

with open(f"{OUT_DIR}/selection.jsonl", "w", encoding="utf-8") as f:
    for r in sel_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

manifest = {
    "dataset": "QuALITY v1.0.1 dev (htmlstripped)",
    "dataset_url": "https://github.com/nyu-mll/quality",
    "paper": "https://arxiv.org/abs/2112.08608",
    "seed": SEED,
    "filters": {"est_tokens_min": MIN_EST_TOKENS, "est_tokens_max": MAX_EST_TOKENS,
                "min_questions": MIN_QUESTIONS, "max_questions_per_passage": MAX_Q_PER_PASSAGE, "n_passages": N_PASSAGES,
                "token_estimate": "len(words)*1.33; must be re-counted with provider endpoint"},
    "stats": {"articles_in_split": len(articles), "candidates_after_filter": len(candidates),
              "selected_passages": len(chosen), "selected_questions": len(sel_rows),
              "non_warmup_requests_per_arm": len(sel_rows) - len(chosen)},
    "passages": manifest_passages,
}
json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"), indent=2)
print(json.dumps(manifest["stats"], indent=2))
