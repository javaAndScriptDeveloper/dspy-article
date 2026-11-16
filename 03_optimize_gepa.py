"""Step 3 — optimize the program with GEPA (Genetic-Pareto reflective evolution).

    uv run python 03_optimize_gepa.py

GEPA keeps a Pareto frontier of instruction candidates, mutates them via an
LLM "reflection LM" that reads textual feedback from the metric, and merges
the best modules from distinct lineages. It tunes *instructions only* — your
program code stays untouched.
"""

import json
import os

import dspy
from dspy.evaluate import Evaluate

from dspy_incident_triage.config import configure, reflection_model
from dspy_incident_triage.data import build_datasets
from dspy_incident_triage.metrics import triage_metric, triage_metric_with_feedback
from dspy_incident_triage.program import IncidentTriageProgram


def main():
    lm = configure()
    trainset, valset = build_datasets()

    program = IncidentTriageProgram()

    optimizer = dspy.GEPA(
        metric=triage_metric_with_feedback,   # note: the feedback-returning metric
        reflection_lm=reflection_model(),     # the "brain" that rewrites instructions
        auto="light",                         # smallest rollout budget
        num_threads=4,
        track_stats=True,                     # fills optimized.detailed_results
        perfect_score=1.25,                   # our metric tops out at 1.25, not GEPA's default 1.0;
                                              # otherwise every right-root-cause example counts as
                                              # "perfect" and GEPA never reflects on severity misses
    )

    print("Compiling with GEPA (reflective prompt evolution on the Pareto frontier)...")
    optimized = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
    )

    evaluate = Evaluate(
        devset=valset,
        metric=triage_metric,
        num_threads=4,
        display_progress=True,
        display_table=3,
    )
    result = evaluate(optimized)
    print(f"\nGEPA-optimized score: {result.score / 100:.2f} per incident, {result.score / 125:.0%} of max (1.25), task model: {lm.model}")

    import json

    os.makedirs("optimized", exist_ok=True)
    results = optimized.detailed_results
    print(f"Candidates explored: {len(results.candidates)}")
    print(f"Best aggregate val score: {max(results.val_aggregate_scores):.3f}")
    with open("optimized/gepa_results.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2, default=str)

    optimized.save("optimized/gepa_triage.json")
    print("Saved instruction to optimized/gepa_triage.json and lineage to optimized/gepa_results.json")

    print("\n--- Evolved instruction (first 400 chars) ---")
    print(optimized.triage.predict.signature.instructions[:400])


if __name__ == "__main__":
    main()
