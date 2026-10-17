# Prefix prompt caching and explicit breakpoints: a reproducible test on QuALITY

Companion repository for the Token Master article. Measures whether Anthropic's
prefix caching behaves as documented, using the provider's own `usage` fields as
ground truth, on a public reading-comprehension dataset.

## Results so far (17 October 2026, Claude Sonnet 5.5, one repetition of arms A and C)

| Metric, 140 measured requests per arm | A cold | C explicit breakpoint |
|---|---|---|
| Cache hit ratio | 0.000 | 0.989 |
| Effective input cost multiplier | 1.00 | 0.110 |
| Mean time to first token | 1,083 ms | 1,051 ms |
| Median time to first token | 1,047 ms | 1,024 ms |
| Accuracy against gold labels | 0.886 | 0.900 |
| Cost for the arm, 160 requests, list prices | $3.17 | $0.80 |

All six pre-registered checks passed. Input cost fell by 89 percent with zero cache
misses and no change in accuracy. Time to first token did not measurably change: the
32 ms difference has a paired bootstrap 95 percent interval of -16 to +81 ms. Arms B,
D and E and repetitions 2 and 3 have not been run yet. Raw per-request rows are in
`phase2/results/rep1.csv`; the three-passage pilot that preceded the full run is in
`phase2/results_pilot/`.

```text
token-master-caching/
├── README.md
├── requirements.txt
├── phase1/
│   ├── prepare_selection.py     # builds the frozen selection from the QuALITY dev split
│   ├── selection.jsonl          # 160 requests: 20 passages x 8 questions (frozen, do not edit)
│   └── manifest.json            # what was selected, passage SHA-256s, provider token counts
└── phase2/
    ├── config.json              # model, pricing, pinned thinking settings, prompt instructions
    ├── config.pilot.json        # same, but writes to results_pilot/ for cheap trial runs
    ├── run_experiment.py        # the harness (count-tokens / dry-run / smoke / run)
    ├── analyze.py               # metrics, pass/fail checks, chart
    ├── make_synthetic_results.py# fake data for testing analyze.py; never publish its output
    ├── results/                 # rep1.csv, summary.csv, checks.txt, chart (real run)
    └── results_pilot/           # smoke.csv and a 3-passage A/C pilot (real, small)
```

## Arms

| Arm | Name | What is different | Expected after warm-up |
|---|---|---|---|
| A | cold | no `cache_control` anywhere | 0 cache reads |
| B | automatic | top-level `cache_control`, provider places the breakpoint | high hit ratio |
| C | explicit | `cache_control` on the passage block | hit ratio ~ passage/(passage+question) |
| D | sabotaged prefix | like C, plus a per-request timestamp above the passage | 0 reads, a write every request |
| E | misplaced breakpoint | `cache_control` on the question block | 0 reads, a write every request, cost multiplier > 1 |

Every (arm, rep) pair embeds its own tag in the instructions block, so prefixes are
distinct across arms and reps. Arms can run back to back without one arm reading
cache entries another arm wrote.

## Requirements

Python 3.10 or newer. The `anthropic` SDK does not install on older interpreters, and
macOS ships 3.9 as `/usr/bin/python3`. Use a virtual environment:

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The API key is read from the `ANTHROPIC_API_KEY` environment variable and nowhere
else. Create a key only for this experiment, export it in the shell you run from, and
never write it into a file in this repository.

```bash
export ANTHROPIC_API_KEY=...
```

## Run it

Approximate costs are at Sonnet 5.5 list prices in `config.json` on 17 October 2026.

```bash
cd phase2
# 0. confirm the model and prices in config.json match the pricing page today

# 1. count tokens with the provider; fills manifest.json, refuses passages under the minimum
python run_experiment.py count-tokens

# 2. look at the exact request body each arm sends (no API call)
python run_experiment.py dry-run --arms A,B,C,D,E

# 3. smoke test, about $0.03: 3 requests, arm C, one passage. #0 should write, #1 and #2 should read.
python run_experiment.py smoke --passages 30035 --config config.pilot.json

# 4. cheap pilot, about $0.45: arms A and C on the three shortest passages, 48 requests
python run_experiment.py run --arms A,C --rep 1 --passages 30035,32665,99911 --config config.pilot.json
python analyze.py --results results_pilot

# 5. one repetition of A and C on all 20 passages, about $4, 320 requests, ~10 minutes
python run_experiment.py run --arms A,C --rep 1

# 6. the remaining arms for the same repetition, about $9
python run_experiment.py run --arms B,D,E --rep 1

# 7. two more repetitions, at least 10 minutes apart so no 5-minute cache entry survives
python run_experiment.py run --arms A,B,C,D,E --rep 2
python run_experiment.py run --arms A,B,C,D,E --rep 3

# 8. metrics, checks, chart
python analyze.py
```

The harness appends to `results/rep<N>.csv` and skips rows already present, so an
interrupted run resumes where it stopped. Rows with an `error` are retried on resume.
The harness stops immediately on a billing or authentication error.

`--passages` takes a comma-separated list of passage ids and `--limit-passages N`
takes the first N of the shuffled order. Both work with `run` and `smoke`.

## What gets logged per request

`cache_creation_input_tokens`, `cache_read_input_tokens`, `input_tokens` (uncached
remainder), `output_tokens`, time to first token (streamed), total latency, the
model's raw answer, the parsed label, the gold label, and the passage SHA-256 so you
can prove the cached prefix was byte-identical.

## Metrics (`analyze.py`)

- hit ratio = cache reads / total input tokens, measured requests only (warm-ups excluded)
- effective input cost multiplier = (writes x 1.25 + reads x 0.10 + uncached x 1.0) / total input
- mean and median time to first token
- accuracy against QuALITY gold labels, re-scored from the raw answer with the current parser
- count of answers that were not a bare option letter
- cost per request with and without warm-ups, at the list prices in `config.json`

Pass criteria are pre-registered in `analyze.py::checks`. Decide them before you run;
do not move them afterwards.

## Model notes for Claude Sonnet 5.5

- `temperature`, `top_p` and `top_k` are not accepted. The 1.x SDK removed the arguments.
- Thinking is on by default. With an 8-token output cap the model would spend the
  budget thinking and never answer, so `config.json` pins
  `thinking: {"type": "between_tools"}`, the only form this model accepts for turning
  thinking off, and pins `effort` explicitly. Every arm sends identical settings; a
  difference would invalidate the cache between arms.
- Minimum cacheable prefix is 512 tokens. `count-tokens` checks every passage.
- The answer parser accepts only a bare option letter, optionally as `B.`, `(B)` or
  `Answer: B`. A sentence that happens to start with a valid letter is scored as a
  non-answer. The first version of the parser credited one such case; the analysis
  re-scores all rows from `raw_answer` so old and new rows agree.

## Things that will silently break the measurement

- Any change to the instructions text, `max_tokens`, thinking or effort settings between
  arms. Thinking parameters invalidate the cache on Anthropic.
- A passage below the model's minimum cacheable prefix. `count-tokens` catches this.
- Gaps longer than 5 minutes between requests for the same passage (default TTL).
- Resuming an interrupted run in the middle of a cached arm (B or C). The passage whose
  warm-up already ran will face a cold cache and show a write where a read was expected.
  Delete that passage's rows for that arm before resuming. Arm A has no cache state and
  resumes safely at any point.
- Trailing whitespace or encoding changes in the passage block. The harness asserts the
  block equals the row's `article` field before every request.

## Costs computed here are not an invoice

Costs are derived from usage counts times the list price in `config.json` on the run
date. Cross-check the totals against the console usage page for the experiment key
before publishing.

## Dataset

QuALITY v1.0.1 dev split, https://github.com/nyu-mll/quality (Pang et al., 2022,
https://arxiv.org/abs/2112.08608). Passages are Project Gutenberg texts in the public
domain. Selection and filters are recorded in `phase1/manifest.json`. The word-count
estimates used for selection ran about 1.6x below the provider's token counts; the
real passages are 6,700 to 12,000 tokens.

## License

Code in this repository is released under the MIT License; see `LICENSE`. The
QuALITY dataset and the Project Gutenberg passages it contains have their own
terms, noted above.
