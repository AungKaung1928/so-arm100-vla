"""End to end on tiny data: record -> train (hash encoder) -> evaluate all
three splits. Proves the plumbing, not the policy. Under 90 s."""
import os

import numpy as np
import pytest

from so_arm100_sim import render
from so_arm100_sim import instructions as I

pytestmark = pytest.mark.skipif(not render.available(), reason="no offscreen GL context")


def test_record_train_eval(tmp_path):
    from so_arm100_sim.datasets import lerobot_available
    if not lerobot_available():
        pytest.skip("lerobot not installed")
    from so_arm100_vla.data import record_multitask
    from train import train
    from eval_language import eval_split, markdown

    root = str(tmp_path / "ds")
    combos = [("reach", "red"), ("reach", "blue")]
    s = record_multitask(root, "test/tiny", episodes_per_combo=2, combos=combos, seed=0,
                         height=32, width=48, log=lambda *a: None)
    assert s["episodes"] == 4 and s["frames"] > 20
    assert set(s["combos"]) == {"reach:red", "reach:blue"}

    model, stats, encoder, trace = train(root, "test/tiny", "hash", k=4, d=32, n_kp=4, layers=1,
                                         heads=4, epochs=2, batch_size=16, lr=1e-3, seed=0,
                                         out=str(tmp_path / "p.pt"), log=lambda *a: None, threads=1)
    assert len(trace) == 2 and np.isfinite(trace[-1]["train_l1"])
    assert os.path.exists(tmp_path / "p.pt") and os.path.exists(tmp_path / "p.json")

    image = dict(camera="front", height=32, width=48)
    results = {}
    for split in ("seen", "paraphrase", "combo"):
        _, split_combos = __import__("so_arm100_vla.splits", fromlist=["x"]).split_spec(split)
        pick = [split_combos[0]]          # any task: the policy is evaluated, not the data
        r = eval_split(model, stats, encoder, split, n_episodes=2, seeds=(0,), image=image,
                       log=lambda *a: None, combos=pick)
        assert len(r["cells"]) == 1 and r["success_rate"] is not None
        cell = next(iter(r["cells"].values()))
        tmpl = "heldout" if split == "paraphrase" else "train"
        task, color = pick[0]
        assert set(cell["sentences_used"]) <= set(I.instructions(task, color, tmpl))
        results[split] = r
    table = markdown(results)
    assert "| seen |" in table and "| combo |" in table
