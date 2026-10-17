"""Phase 2 analysis: metrics per arm, pass/fail checks, and the centerpiece chart.

Usage
  python analyze.py                      # reads results/rep*.csv
  python analyze.py --results results    # explicit dir

Outputs (in the results dir)
  summary.csv          per (arm, rep) metrics
  summary_by_arm.csv   mean and spread across reps
  checks.txt           pass/fail against the pre-registered criteria
  cached_tokens_by_request_index.png
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_ORDER = ["A", "B", "C", "D", "E"]
ARM_LABEL = {"A": "A cold", "B": "B automatic", "C": "C explicit", "D": "D sabotaged prefix", "E": "E misplaced breakpoint"}
# fixed categorical slots (dataviz reference palette, light surface)
ARM_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}


def load(results_dir: Path, cfg: dict) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "rep*.csv")))
    if not files:
        raise SystemExit(f"no rep*.csv in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    # Re-score from the raw answer with the current parser so old and new rows agree.
    sys.path.insert(0, str(Path(__file__).parent))
    from run_experiment import parse_answer
    df["pred_label"] = df["raw_answer"].map(parse_answer)
    df["correct"] = (df["pred_label"] == pd.to_numeric(df["gold_label"], errors="coerce")).astype(int)
    df["non_letter_answer"] = df["pred_label"].isna().astype(int)
    for c in ["cache_creation_input_tokens", "cache_read_input_tokens", "input_tokens", "output_tokens",
              "total_input_tokens", "ttft_ms", "latency_ms", "correct", "is_warmup", "request_index_in_passage", "rep"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    m = cfg["pricing_multipliers"]
    df["effective_input_units"] = (df["cache_creation_input_tokens"] * m["cache_write_5m"]
                                   + df["cache_read_input_tokens"] * m["cache_read"]
                                   + df["input_tokens"] * m["uncached_input"])
    df["input_cost_usd"] = df["effective_input_units"] / 1e6 * cfg["base_input_price_per_mtok_usd"]
    df["output_cost_usd"] = df["output_tokens"] / 1e6 * cfg["base_output_price_per_mtok_usd"]
    df["cost_usd"] = df["input_cost_usd"] + df["output_cost_usd"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    meas = df[df["is_warmup"] == 0]
    g = meas.groupby(["arm", "rep"])
    out = pd.DataFrame({
        "requests": g.size(),
        "hit_ratio": g["cache_read_input_tokens"].sum() / g["total_input_tokens"].sum(),
        "write_ratio": g["cache_creation_input_tokens"].sum() / g["total_input_tokens"].sum(),
        "effective_cost_multiplier": g["effective_input_units"].sum() / g["total_input_tokens"].sum(),
        "mean_ttft_ms": g["ttft_ms"].mean(),
        "p50_ttft_ms": g["ttft_ms"].median(),
        "accuracy": g["correct"].mean(),
        "non_letter_answers": g["non_letter_answer"].sum(),
        "mean_cost_per_request_usd": g["cost_usd"].mean(),
    }).reset_index()
    # cost per task including warm-ups, since production pays for them too
    all_g = df.groupby(["arm", "rep"])["cost_usd"].sum() / df.groupby(["arm", "rep"]).size()
    out = out.merge(all_g.rename("cost_per_request_incl_warmup_usd").reset_index(), on=["arm", "rep"])
    return out.sort_values(["arm", "rep"])


def by_arm(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in summary.columns if c not in ("arm", "rep", "requests")]
    g = summary.groupby("arm")[cols]
    agg = g.mean().add_suffix("_mean").join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max"))
    agg["reps"] = summary.groupby("arm")["rep"].nunique()
    return agg.reindex([a for a in ARM_ORDER if a in agg.index])


def checks(summary: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """Pre-registered pass criteria. Edit here only before you run, never after."""
    lines = []
    def arm(a): return summary[summary["arm"] == a]
    def ok(cond, msg): lines.append(("PASS  " if cond else "FAIL  ") + msg)

    if len(arm("C")):
        ok((arm("C")["hit_ratio"] >= 0.85).all(), "C: hit ratio >= 0.85 on every rep")
        ok((arm("C")["effective_cost_multiplier"] <= 0.35).all(), "C: effective input cost multiplier <= 0.35")
    for a in ("A", "D"):
        if len(arm(a)):
            ok((arm(a)["hit_ratio"] == 0).all(), f"{a}: zero cache reads on every rep")
    if len(arm("E")):
        ok((arm("E")["hit_ratio"] == 0).all(), "E: zero cache reads (breakpoint on changing content)")
        ok((arm("E")["effective_cost_multiplier"] > 1.0).all(), "E: effective cost multiplier > 1.0 (paying for writes never read)")
    if len(arm("A")) and len(arm("C")):
        ok(arm("C")["mean_ttft_ms"].mean() < arm("A")["mean_ttft_ms"].mean(), "C: mean TTFT lower than A")
        acc = summary.groupby("arm")["accuracy"].mean()
        spread = acc.reindex([a for a in ("A", "B", "C") if a in acc.index])
        ok(spread.max() - spread.min() <= 0.05, f"A/B/C accuracy within 5 points of each other ({spread.round(3).to_dict()})")
    # per-passage sanity: in C, every non-warmup request should read the passage
    c = df[(df["arm"] == "C") & (df["is_warmup"] == 0)]
    if len(c):
        misses = c[c["cache_read_input_tokens"] == 0]
        ok(len(misses) == 0, f"C: no non-warmup request with zero cache reads ({len(misses)} misses of {len(c)})")
    return lines


def chart(df: pd.DataFrame, out_png: Path) -> None:
    """Mean cached-read tokens vs request index within a passage, one line per arm."""
    d = df.groupby(["arm", "request_index_in_passage"])["cache_read_input_tokens"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    # arms that behave identically overlap exactly, so each gets its own dash pattern and marker
    style = {"A": ("-", "o"), "B": ("--", "s"), "C": ("-", "^"), "D": (":", "D"), "E": ("-.", "v")}
    for a in ARM_ORDER:
        s = d[d["arm"] == a]
        if s.empty:
            continue
        ls, mk = style[a]
        ax.plot(s["request_index_in_passage"], s["cache_read_input_tokens"], color=ARM_COLOR[a], lw=2, ls=ls,
                marker=mk, ms=6, markerfacecolor="white", markeredgewidth=1.6, label=ARM_LABEL[a])
    ax.set_xlabel("request index within passage (0 = warm-up)")
    ax.set_ylabel("cached input tokens read (mean over passages)")
    ax.set_title("Prefix caching on QuALITY: tokens served from cache per request", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    ax.set_xticks(sorted(d["request_index_in_passage"].unique()))
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    results = Path(a.results)
    df = load(results, cfg)
    summary = summarize(df)
    summary.to_csv(results / "summary.csv", index=False)
    arm_tbl = by_arm(summary)
    arm_tbl.to_csv(results / "summary_by_arm.csv")
    lines = checks(summary, df)
    (results / "checks.txt").write_text("\n".join(lines) + "\n")
    chart(df, results / "cached_tokens_by_request_index.png")

    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\nPer (arm, rep):\n", summary.to_string(index=False))
    print("\nBy arm (mean / min / max across reps):\n",
          arm_tbl[["hit_ratio_mean", "hit_ratio_min", "hit_ratio_max", "effective_cost_multiplier_mean",
                   "mean_ttft_ms_mean", "accuracy_mean", "accuracy_min", "accuracy_max", "reps"]].to_string())
    print("\nChecks:\n" + "\n".join(lines))
    print(f"\nwrote summary.csv, summary_by_arm.csv, checks.txt, cached_tokens_by_request_index.png in {results}")


if __name__ == "__main__":
    main()
