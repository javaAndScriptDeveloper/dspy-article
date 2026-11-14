"""Step 1 — evaluate the UNOPTIMIZED program.

    uv run python 01_baseline.py

You get the baseline score that the optimizers have to beat.
"""

from dspy.evaluate import Evaluate

from dspy_incident_triage.config import configure
from dspy_incident_triage.data import build_datasets
from dspy_incident_triage.metrics import triage_metric
from dspy_incident_triage.program import IncidentTriageProgram


def main():
    configure()
    trainset, valset = build_datasets()

    program = IncidentTriageProgram()

    evaluate = Evaluate(
        devset=valset,
        metric=triage_metric,
        num_threads=4,
        display_progress=True,
        display_table=3,
    )
    result = evaluate(program)
    print(f"\nBaseline score on {len(valset)} incidents: {result.score:.3f} (max 1.25)")


if __name__ == "__main__":
    main()
