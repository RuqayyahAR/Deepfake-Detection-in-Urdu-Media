"""
Detector interface for evaluation (step 2).

Any detector must implement score(path) -> float, where higher = more likely FAKE.
A pretrained model is plugged in by writing a small class like PretrainedDetector
below and passing it as --detector module:Class to run_eval.py.
"""
import importlib
import os
import sys


class Detector:
    name = "base"

    def score(self, path):
        raise NotImplementedError


class HeuristicDetector(Detector):
    """Baseline: the existing TRACE heuristic engine. Weak; kept as a reference point."""

    name = "heuristic-trace-engine"

    def __init__(self, engine_dir):
        sys.path.insert(0, engine_dir)
        from deepfake_engine import DeepfakeEngine

        self.engine = DeepfakeEngine()

    def score(self, path):
        return float(self.engine.analyze(path)["suspicion_score"])


class PretrainedDetector(Detector):
    """TEMPLATE - not runnable until a model is chosen.

    Steps: 1) choose the model and check its licence;
    2) download weights on a machine with internet access;
    3) fill in __init__ and score(). Keep score() returning a probability of FAKE
    for ONE video (average over sampled frames or clips).
    """

    name = "pretrained-TODO"

    def __init__(self, weights_path):
        raise NotImplementedError("No pretrained model has been selected yet.")

    def score(self, path):
        raise NotImplementedError


def load(spec, engine_dir):
    if spec == "heuristic":
        return HeuristicDetector(engine_dir)
    module, cls = spec.split(":")
    return getattr(importlib.import_module(module), cls)()
