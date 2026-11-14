"""The DSPy program: Spring Boot incident triage.

This is the whole "model". Notice what is NOT here: no prompt text, no
few-shot examples, no model name. Those are the *parameters* the optimizers
(MIPROv2 / GEPA) will tune.
"""

from typing import Literal

import dspy

from .data import ROOT_CAUSES


class IncidentTriage(dspy.Signature):
    """Classify a Spring Boot production incident by its root cause and severity, and propose a fix."""

    incident_report: str = dspy.InputField(
        desc="Log line, stack trace excerpt or error description from a running Spring Boot service"
    )
    root_cause: Literal[tuple(ROOT_CAUSES)] = dspy.OutputField(  # noqa: FURB123
        desc="One of the fixed root-cause categories"
    )
    severity: Literal["P1", "P2", "P3"] = dspy.OutputField(
        desc="P1 = service down / data risk, P2 = degraded, P3 = cosmetic or contained"
    )
    action: str = dspy.OutputField(desc="One-sentence remediation a senior engineer should apply")


class IncidentTriageProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.triage = dspy.ChainOfThought(IncidentTriage)

    def forward(self, incident_report: str) -> dspy.Prediction:
        return self.triage(incident_report=incident_report)
