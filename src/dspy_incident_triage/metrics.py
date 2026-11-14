"""Metrics for evaluating and optimizing the triage program.

Two flavors:
- ``triage_metric`` — a plain scalar metric. Good enough for MIPROv2 and for
  dspy.Evaluate.
- ``triage_metric_with_feedback`` — GEPA wants *textual* feedback explaining
  WHY a prediction is wrong, so the reflection LM can propose a better
  instruction. This is the whole trick behind reflective prompt evolution.
"""

import dspy

from .data import ROOT_CAUSES


def triage_metric(example, pred, trace=None) -> float:
    """Plain scalar metric: root cause is what we care about, severity is a bonus."""
    root_ok = pred.root_cause == example.root_cause
    sev_ok = pred.severity == example.severity
    return (1.0 if root_ok else 0.0) + (0.25 if sev_ok else 0.0)


_SEVERITY_HINTS = {
    "P1": "P1 means the service is down or data is at risk",
    "P2": "P2 means degraded behavior, but the service still responds",
    "P3": "P3 means cosmetic or contained impact",
}


def triage_metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-style metric: returns dspy.Prediction(score, feedback).

    GEPA feeds ``feedback`` verbatim into the reflection LM. A bare float is
    accepted too, but then GEPA is "a much weaker version of itself" (dspy docs).
    """
    problems = []

    if pred.root_cause == example.root_cause:
        root_score = 1.0
    else:
        root_score = 0.0
        problems.append(
            f"Wrong root cause: you said '{pred.root_cause}', the correct label is "
            f"'{example.root_cause}'. Categories are: {', '.join(ROOT_CAUSES)}."
        )

    if pred.severity == example.severity:
        sev_score = 0.25
    else:
        sev_score = 0.0
        problems.append(
            f"Wrong severity: you said '{pred.severity}', expected '{example.severity}'. "
            f"{_SEVERITY_HINTS[example.severity]}."
        )

    score = root_score + sev_score
    feedback = " ".join(problems) if problems else "Correct root cause and severity."
    return dspy.Prediction(score=score, feedback=feedback)
