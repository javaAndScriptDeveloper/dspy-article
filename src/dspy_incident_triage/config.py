"""Shared setup: read the LM model from the environment and configure DSPy."""

import os

import dspy


def task_model() -> dspy.LM:
    """The cheap model the triage program runs on (and that gets optimized)."""
    model = os.environ.get("DSPY_MODEL", "gemini/gemini-3.5-flash-lite")
    return dspy.LM(model, max_tokens=1024, temperature=0.7, num_retries=6)


def reflection_model() -> dspy.LM:
    """The strong model GEPA uses to *reflect* and propose new instructions.

    GEPA runs a handful of reflection calls (serial), so a stronger model here
    is worth it — but it is the expensive knob, so override via DSPY_REFLECTION_MODEL
    to use another provider. num_retries rides out 429s on free-tier quotas.
    """
    model = os.environ.get("DSPY_REFLECTION_MODEL", "gemini/gemini-3.5-flash")
    return dspy.LM(model, max_tokens=8192, temperature=1.0, num_retries=6)


def configure() -> dspy.LM:
    if os.environ.get("DSPY_CACHE") == "0":
        # DSPy caches LM responses on disk: identical reruns return identical answers.
        # Disable it to measure real run-to-run variance.
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    lm = task_model()
    dspy.configure(lm=lm)
    return lm
