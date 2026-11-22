"""The split summaries in runs/eval_*.json are recomputed from their own cells.

README rows are checked against the summary numbers (test_readme.py); this checks
that those summary numbers are what summarise_split makes of the per-cell results
stored next to them, so neither the aggregation code nor a hand-edited JSON can drift.
"""
import glob
import json
import os

import pytest

E = pytest.importorskip("eval_language")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = sorted(glob.glob(os.path.join(ROOT, "runs", "eval_policy_*_*.json")))


def test_there_are_tracked_eval_files():
    assert len(FILES) == 9, FILES        # 3 policies x 3 splits


@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(f) for f in FILES])
def test_summary_equals_summarise_split_of_its_cells(path):
    r = json.load(open(path))
    s = E.summarise_split(r["split"], r["templates"], r["cells"], r["n_episodes"], r["seeds"], r["physics"])
    assert s["success_rate"] == r["success_rate"]
    assert s["success_std_mean"] == r["success_std_mean"]
    assert s["by_task"] == r["by_task"]
