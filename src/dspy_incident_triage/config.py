"""Shared setup: read the LM model from the environment and configure DSPy."""

import os

import dspy


def task_model() -> dspy.LM:
    """The cheap model the triage program runs on (and that gets optimized)."""
    model = os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini")
    return dspy.LM(model, max_tokens=1024, temperature=0.7)


def reflection_model() -> dspy.LM:
    """The strong model GEPA uses to *reflect* and propose new instructions.

    GEPA runs a handful of reflection calls (serial), so a stronger model here
    is worth it — but it is the expensive knob, so override via DSPY_REFLECTION_MODEL
    if you use a non-OpenAI provider.
    """
    model = os.environ.get("DSPY_REFLECTION_MODEL", "openai/gpt-4o")
    return dspy.LM(model, max_tokens=8192, temperature=1.0)


def configure() -> dspy.LM:
    lm = task_model()
    dspy.configure(lm=lm)
    return lm
