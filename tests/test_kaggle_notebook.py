"""The Kaggle notebook is generated; check it builds, parses, and records only training pairs."""
import ast
import importlib.util
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("make_notebook",
                                              os.path.join(HERE, "..", "kaggle", "make_notebook.py"))
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)


def test_python_cells_parse():
    nb = mk.build()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    py = [c for c in code if not c.startswith("%%bash")]
    assert len(py) >= 4
    for src in py:
        ast.parse(src)


def test_records_exactly_the_training_pairs():
    I = pytest.importorskip("so_arm100_sim.instructions")
    specs = mk.COMBOS.split()
    assert all(s.endswith(":distractors") for s in specs)   # same scenes as data.py and the bench
    combos = [tuple(s.split(":")[:2]) for s in specs]
    assert combos == list(I.TRAIN_COMBOS)
    assert not set(combos) & set(I.HELDOUT_COMBOS)


def _train_cell():
    return next("".join(c["source"]) for c in mk.build()["cells"]
                if "def s_per_step" in "".join(c["source"]))


def test_s_per_step_is_update_plus_data_wait_after_warmup():
    ns = {}
    exec(_train_cell().split("for BATCH")[0], ns)   # defs only, no training
    log = ("updt_s:80.0 data_s:5.0 x updt_s:9.0 data_s:1.0 y "
           "updt_s:1.0 data_s:0.5 z updt_s:3.0 data_s:1.5")
    assert ns["s_per_step"](log) == pytest.approx(3.0)


def test_s_per_step_reads_a_real_lerobot_log_line():
    lu = pytest.importorskip("lerobot.utils.logging_utils")
    meters = {"update_s": lu.AverageMeter("updt_s", ":.3f"), "dataloading_s": lu.AverageMeter("data_s", ":.3f")}
    tracker = lu.MetricsTracker(8, 100, 10, meters)
    lines = []
    for u, d in ((9.0, 9.0), (9.0, 9.0), (0.7, 0.1), (0.9, 0.3)):
        tracker.update_s, tracker.dataloading_s = u, d
        lines.append(str(tracker))
        tracker.reset_averages()
    ns = {}
    exec(_train_cell().split("for BATCH")[0], ns)
    assert ns["s_per_step"]("\n".join(lines)) == pytest.approx(1.0)


def test_only_the_zip_and_logs_land_in_the_saved_output_folder():
    code = "".join("".join(c["source"]) for c in mk.build()["cells"] if c["cell_type"] == "code")
    saved = set(re.findall(r"/kaggle/working/[\w./]+", code))
    assert saved == {"/kaggle/working/smolvla_sim.zip", "/kaggle/working/train.log",
                     "/kaggle/working/timing.log"}   # venv, data, checkpoints: /kaggle/temp (20 GB cap)


def test_committed_notebook_is_the_generated_one():
    with open(os.path.join(HERE, "..", "kaggle", "smolvla_finetune.ipynb")) as f:
        assert json.load(f) == mk.build()
