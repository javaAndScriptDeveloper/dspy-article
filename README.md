# DSPy Incident Triage — GEPA vs MIPROv2, a runnable example

Companion code for the article
**[DSPy Explained: GEPA vs MIPROv2](https://codewithkyryl.dev/2026/11/14/dspy-gepa-vs-miprov2/)**
on codewithkyryl.dev.

The task: given a Spring Boot production incident (a log line or stack-trace
excerpt), classify the **root cause** (6 categories), the **severity**
(P1/P2/P3) and propose a one-sentence fix. Small enough to run in minutes,
real enough to show what prompt optimizers actually do.

The whole "model" is one DSPy module (`src/dspy_incident_triage/program.py`)
— **no prompt text, no few-shot examples, no model name in the source**.
Those are the parameters the optimizers tune.

## Run it

```bash
uv sync                                   # installs dspy (Python 3.11+)
uv run python smoke_test.py               # offline check — no API key needed
```

The smoke test drives the entire pipeline (program forward, both metrics,
MIPROv2 constructor, a full GEPA compile, save/load) against DSPy's `DummyLM`,
so it validates the wiring without spending a cent.

For the real runs you need an LLM API key:

```bash
export GEMINI_API_KEY=...                 # free key: https://aistudio.google.com/apikey

uv run python 01_baseline.py              # score the unoptimized program
uv run python 02_optimize_miprov2.py      # Bayesian joint search (instructions + demos)
uv run python 03_optimize_gepa.py         # reflective prompt evolution (Pareto frontier)
```

Defaults: task model `gemini/gemini-3.5-flash-lite` (override with `DSPY_MODEL`),
GEPA reflection model `gemini/gemini-3.5-flash` (override with `DSPY_REFLECTION_MODEL`).
Any `dspy.LM` string works, e.g. `openai/gpt-4o-mini` with `OPENAI_API_KEY`.
Free-tier quotas are small and change; check your project's limits in AI Studio
before the optimizer runs, which make hundreds of calls.
The dataset is 24 examples, so a run is small.

## What I measured (17 Nov 2026, Gemini free tier, `auto="light"`)

Per-incident score on the 10 validation incidents (max 1.25):

| Run | Score |
|---|---|
| Unoptimized, 3 uncached runs | 1.10 / 1.12 / 1.10 |
| MIPROv2 | 1.12 (within noise) |
| GEPA, default `perfect_score` | 1.10 (no reflection happens, see below) |
| GEPA, `perfect_score=1.25` | 1.175 (21 candidates, 420 metric calls, ~18 min) |

Treat these as a wiring demo, not a benchmark: 10 validation examples, and GEPA
selects its winner on the same set. Gotchas the run exposed:

- GEPA's `perfect_score` defaults to 1.0; this metric maxes at 1.25, so without
  `perfect_score=1.25` GEPA never reflects on severity misses.
- DSPy caches LM responses on disk. `DSPY_CACHE=0` disables it for real variance.
- `MIPROv2.compile()` needs `optuna` (`dspy[optuna]` is in `pyproject.toml`).
- Read `optimized/gepa_triage.json`: the evolved instruction memorizes training
  incidents and invents a root cause outside the allowed six.

## What the optimizers change

Both write their result into `optimized/`:

- `02_optimize_miprov2.py` → `optimized/miprov2_triage.json` — learned
  **instructions + few-shot demos**.
- `03_optimize_gepa.py` → `optimized/gepa_triage.json` — the evolved
  **instruction** (GEPA does not touch demos), plus
  `optimized/gepa_results.json` with the full lineage: every candidate,
  its parents and per-example validation scores.

Diff `optimized/gepa_triage.json` against the source program to see what the
optimizer actually wrote into your prompt — that's the DSPy pitch in one file.

## Layout

```
src/dspy_incident_triage/
├── data.py        # 24 hand-crafted incidents (14 train / 10 val)
├── program.py     # Signature + ChainOfThought module — the whole "model"
├── metrics.py     # scalar metric + GEPA feedback metric (score + WHY it's wrong)
└── config.py      # LM configuration via env vars
01_baseline.py         # evaluate the unoptimized program
02_optimize_miprov2.py # dspy.MIPROv2(metric, auto="light") + compile + evaluate
03_optimize_gepa.py    # dspy.GEPA(metric+feedback, reflection_lm) + compile + evaluate
smoke_test.py          # offline verification, no API key
```
