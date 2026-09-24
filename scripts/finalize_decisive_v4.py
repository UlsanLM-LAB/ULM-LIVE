#!/usr/bin/env python3
"""Summarize the matched remote continuation and draw first-error diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.pilot_talker_ar_v3 import write_json  # noqa: E402

ROOT = Path("outputs/talker-decisive-v4")


def main():
    base = json.loads((ROOT / "2048-cont-base/final.json").read_text())
    tf = json.loads((ROOT / "2048-cont-tf/history.json").read_text())
    rollout = json.loads((ROOT / "2048-cont-rollout/history.json").read_text())
    if [x["step"] for x in tf] != [x["step"] for x in rollout]:
        raise ValueError("branches have unmatched evaluation steps")
    if tf[-1]["step"] != 16000:
        raise ValueError("continuations incomplete")
    selection = json.loads((ROOT / "selection.json").read_text())
    if set(selection["train_ids"]) & set(selection["val_tf_ids"]):
        raise ValueError("validation overlaps training")
    final_feedback = rollout[-1]["feedback"]
    if not (.47 <= final_feedback["effective_feedback_ratio"] <= .53):
        raise ValueError("final effective feedback misses 50% ±3pp")
    if final_feedback["exposed_sample_fraction"] <= .9:
        raise ValueError("fewer than 90% of samples exposed to learner history")
    listening = json.loads((ROOT / "listening_summary.json").read_text())
    compact = {"git_base_head": "be23839a41268500e182e6262558e99213ab5a6d",
               "base_checkpoint": "outputs/talker-ar-diagnostics-v3/pilots-postfix/2048-multi-baseline/final.pt",
               "validation_count": len(selection["val_tf_ids"]),
               "ar_validation_count": len(selection["val_ar_ids"]),
               "base": base, "tf": tf, "rollout": rollout,
               "listening": listening, "asr": "ASR NOT AVAILABLE LOCALLY"}
    write_json(ROOT / "final_summary.json", compact)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for label, item, color in (("TF continuation", tf[-1], "#295480"),
                                ("Rollout continuation", rollout[-1], "#ba5b36")):
        s = item["first_error"]["survival"]
        xs = [int(x) for x in s]
        axes[0].plot(xs, [s[str(x)] for x in xs], "o-", label=label, color=color)
        p = item["first_error"]["accuracy_after_first_error"]
        axes[1].plot([int(x) for x in p], [p[x]["mean"] for x in p], "o-",
                     label=label, color=color)
    axes[0].set(title="No AR error through frame t", xlabel="Frame t", ylabel="Probability",
                ylim=(-.03, 1.03))
    axes[1].set(title="Codec accuracy after first wrong frame", xlabel="Frames after error",
                ylabel="Mean exact token accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.savefig(ROOT / "first_error_survival.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
