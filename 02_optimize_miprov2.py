"""Step 2 — optimize the program with MIPROv2.

    uv run python 02_optimize_miprov2.py

MIPROv2 jointly proposes instruction candidates AND few-shot demo sets, then
picks the best combination with Bayesian optimization over minibatch evals.
"""

import os

import dspy
from dspy.evaluate import Evaluate

from dspy_incident_triage.config import configure
from dspy_incident_triage.data import build_datasets
from dspy_incident_triage.metrics import triage_metric
from dspy_incident_triage.program import IncidentTriageProgram


def main():
    lm = configure()
    trainset, valset = build_datasets()

    program = IncidentTriageProgram()

    optimizer = dspy.MIPROv2(
        metric=triage_metric,
        auto="light",          # 6 candidates — keep the demo cheap
        num_threads=4,
    )

    print("Compiling with MIPROv2 (bootstrapping demos + Bayesian search)...")
    optimized = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        requires_permission_to_run=False,  # for CI / scripts; interactive default is True
    )

    evaluate = Evaluate(
        devset=valset,
        metric=triage_metric,
        num_threads=4,
        display_progress=True,
        display_table=3,
    )
    result = evaluate(optimized)
    print(f"\nMIPROv2-optimized score: {result.score / 100:.2f} per incident, {result.score / 125:.0%} of max (1.25), task model: {lm.model}")

    os.makedirs("optimized", exist_ok=True)
    optimized.save("optimized/miprov2_triage.json")
    print("Saved instruction + demos to optimized/miprov2_triage.json")

    # What did the optimizer actually write into our prompt?
    print("\n--- Learned instruction (first 400 chars) ---")
    print(optimized.triage.predict.signature.instructions[:400])


if __name__ == "__main__":
    main()
