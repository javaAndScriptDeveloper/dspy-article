"""Offline smoke test — verifies the whole example WITHOUT any API key.

DSPy ships a DummyLM that plays back canned responses, so we can exercise the
program, both metrics, and even a full GEPA optimization loop locally.

    uv run python smoke_test.py
"""

import sys

import dspy
from dspy.utils import DummyLM

from dspy_incident_triage.data import ROOT_CAUSES, build_datasets
from dspy_incident_triage.metrics import triage_metric, triage_metric_with_feedback
from dspy_incident_triage.program import IncidentTriageProgram

VALID_ANSWER = {
    "plan": "Read the log line and match the stack trace signature.",
    "root_cause": "connection-pool-exhaustion",
    "severity": "P1",
    "action": "Raise pool size and fix the leaking transaction.",
}


def canned_lm(n=500):
    """A DummyLM that always returns a structurally valid triage."""
    return DummyLM([dict(VALID_ANSWER)] * n)


def test_dataset():
    trainset, valset = build_datasets()
    assert len(trainset) == 14 and len(valset) == 10, "expected 14 train / 10 val"
    for ex in trainset + valset:
        assert ex.root_cause in ROOT_CAUSES
        assert ex.severity in {"P1", "P2", "P3"}
        assert list(ex.inputs().keys()) == ["incident_report"]
    print(f"OK  dataset: {len(trainset)} train / {len(valset)} val, labels valid")


def test_program_forward():
    dspy.configure(lm=canned_lm())
    out = IncidentTriageProgram()(incident_report="HikariPool-1 timed out after 30000ms")
    assert out.root_cause == "connection-pool-exhaustion"
    assert out.severity == "P1"
    assert "pool" in out.action.lower()
    assert "reasoning" in out.keys(), "ChainOfThought should add a reasoning field"
    print("OK  program forward (ChainOfThought) parses root_cause/severity/action")


def test_metrics():
    trainset, _ = build_datasets()
    ex = trainset[0]  # config-misconfiguration / P1

    class GoodPred:
        root_cause = "config-misconfiguration"
        severity = "P1"

    class WrongPred:
        root_cause = "thread-starvation"
        severity = "P3"

    assert triage_metric(ex, GoodPred()) == 1.25
    assert triage_metric(ex, WrongPred()) == 0.0

    gepa_good = triage_metric_with_feedback(ex, GoodPred())
    assert isinstance(gepa_good, dspy.Prediction) and gepa_good.score == 1.25

    gepa_bad = triage_metric_with_feedback(ex, WrongPred())
    assert gepa_bad.score == 0.0
    assert "thread-starvation" in gepa_bad.feedback, "feedback must explain the mistake"
    print("OK  scalar metric + GEPA feedback metric behave correctly")


def test_miprov2_constructor():
    dspy.configure(lm=canned_lm())  # MIPROv2 needs a default LM at construction
    opt = dspy.MIPROv2(metric=triage_metric, auto="light", num_threads=2)
    assert opt is not None
    print("OK  MIPROv2 constructor (auto='light')")


def test_gepa_end_to_end_offline():
    """The real thing: a full GEPA reflective-evolution loop, just with a DummyLM."""
    reflection_lm = DummyLM(
        [
            "Improved instruction: first match the stack trace signature "
            "(Hikari, Jackson, OOM, thread pool), then map the blast radius to P1-P3.",
            "none",
        ]
    )
    dspy.configure(lm=canned_lm())

    trainset, valset = build_datasets()
    opt = dspy.GEPA(
        metric=triage_metric_with_feedback,
        reflection_lm=reflection_lm,
        max_metric_calls=40,  # tiny explicit budget — smoke run only
        num_threads=2,
        track_stats=True,
    )
    optimized = opt.compile(IncidentTriageProgram(), trainset=trainset, valset=valset)

    results = optimized.detailed_results
    assert len(results.candidates) >= 1
    # The winner's instruction lives on the predictor: module.predict.signature
    instruction = optimized.triage.predict.signature.instructions
    assert instruction, "compiled program must carry an instruction"
    print(f"OK  GEPA compile ran end-to-end offline ({len(results.candidates)} candidates)")


def test_save_load():
    dspy.configure(lm=canned_lm())
    optimized = IncidentTriageProgram()
    optimized.save("/tmp/dspy_smoke_triage.json")
    loaded = IncidentTriageProgram()
    loaded.load("/tmp/dspy_smoke_triage.json")
    assert loaded.triage.predict.signature.instructions == \
        optimized.triage.predict.signature.instructions
    print("OK  program save()/load() round-trip")


if __name__ == "__main__":
    test_dataset()
    test_program_forward()
    test_metrics()
    test_miprov2_constructor()
    test_gepa_end_to_end_offline()
    test_save_load()
    print("\nAll smoke tests passed — the example is wired correctly.")
    print("Set OPENAI_API_KEY and run 01_baseline.py, 02_optimize_miprov2.py, 03_optimize_gepa.py for real.")
    sys.exit(0)
