"""Checks for diagnostic measurements that determine the next training gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_talker_ar_v3.py"
SPEC = importlib.util.spec_from_file_location("diagnose_talker_ar_v3", SCRIPT)
assert SPEC and SPEC.loader
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def test_first_divergence_and_q0_track_different_errors() -> None:
    target = torch.tensor([[1, 2, 3], [4, 5, 6]])
    predicted = torch.tensor([[1, 2, 7], [4, 9, 6]])
    assert diagnostics.first_divergence(predicted, target) == 1
    assert diagnostics.first_q0_divergence(predicted, target) == 2
    assert diagnostics.first_divergence(target, target) is None


def test_stop_frame_honors_minimum_and_threshold() -> None:
    assert diagnostics.stop_frame([0.8, 0.2, 0.6], minimum=2) == 3
    assert diagnostics.stop_frame([0.1, 0.2], minimum=1) == 2
