"""Every README number that has a tracked runs/*.json is recomputed from that file.

The edge tables (OpenVINO, INT8, camera and step sweeps) have no tracked JSON and are not covered.
"""
import json
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
SPLITS = ("seen", "paraphrase", "combo")
ROWS = {"minilm": "`LangPolicy`, MiniLM encoder, FiLM", "hash": "`LangPolicy`, hashed bag-of-words encoder",
        "minilm_nofilm": "`LangPolicy`, MiniLM, no FiLM"}


def run(name):
    with open(os.path.join(ROOT, "runs", name + ".json")) as f:
        return json.load(f)


def cell(r):
    return f"{r['success_rate']:.3f} ± {r['success_std_mean']:.3f}"


@pytest.mark.parametrize("tag", ROWS)
def test_at_a_glance_row(tag):
    cells = " | ".join(cell(run(f"eval_policy_{tag}_s0_{s}")) for s in SPLITS)
    assert f"| {ROWS[tag]} | {cells} |" in README


@pytest.mark.parametrize("split", SPLITS)
def test_minilm_results_row_and_per_task(split):
    r = run(f"eval_policy_minilm_s0_{split}")
    line = next(l for l in README.splitlines()
                if l.startswith(f"| `LangPolicy`, MiniLM | success, `{split}` |"))
    assert f"| {cell(r)} |" in line
    assert (r["n_episodes"], len(r["seeds"])) == (20, 5)      # "20 episodes × 5 evaluation seeds"
    if split != "combo":
        want = ", ".join(f"{t} {r['by_task'][t]:.2f}" for t in ("reach", "push", "lift", "pick_place"))
        assert want in line


def test_smolvla_zero_shot_line():
    r = run("smolvla_zero_shot")
    wins = sum(round(c["success_rate"] * c["n_episodes"]) for c in r["cells"].values())
    total = sum(c["n_episodes"] for c in r["cells"].values())
    assert f"{wins} of {total} episodes" in README
    assert (f"| {r['forward_s_p50']:.2f} s (p90 {r['forward_s_p90']:.2f} s, n = {r['n_forward']}) |"
            in README)


def test_lora_gate_line():
    r = run("lora_gate")
    line = next(l for l in README.splitlines() if l.startswith("| LoRA gate"))
    share = 100 * r["params_trainable"] / r["params_total"]
    for want in (f"**{r['step_s_best']:.2f} s, best of {len(r['step_s'])} → {r['verdict']}**",
                 f"GO under {r['gate_s']:.0f} s/step", f"{r['params_total'] / 1e6:.1f}M stand-in",
                 f"{r['params_trainable'] / 1e6:.2f}M trainable ({share:.2f}%)",
                 f"batch 1, seq {r['seq']}, {r['threads']} threads"):
        assert want in line, want
