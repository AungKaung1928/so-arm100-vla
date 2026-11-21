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


def test_zip_cell_packs_the_checkpoint_behind_the_last_symlink(tmp_path):
    import zipfile
    out, ds, work = tmp_path / "train", tmp_path / "ds", tmp_path / "working"
    model = out / "checkpoints" / "000020" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"w" * 10)
    (model / "config.json").write_text("{}")
    (out / "checkpoints" / "000020" / "training_state").mkdir()
    (out / "checkpoints" / "000020" / "training_state" / "optimizer.safetensors").write_bytes(b"o")
    (out / "checkpoints" / "last").symlink_to("000020")          # as lerobot's update_last_checkpoint
    ds.mkdir(); (ds / "recording_log.json").write_text("{}")
    work.mkdir(); (work / "train.log").write_text("step:1")
    src = next("".join(c["source"]) for c in mk.build()["cells"] if "zipfile" in "".join(c["source"]))
    exec(src.replace("/kaggle/working", str(work)), {"os": os, "OUT": str(out), "DS": str(ds)})
    names = set(zipfile.ZipFile(work / "smolvla_sim.zip").namelist())
    assert names == {"train/checkpoints/last/pretrained_model/model.safetensors",
                     "train/checkpoints/last/pretrained_model/config.json",
                     "train/recording_log.json", "train/train.log"}   # no optimizer state


def test_timing_cell_falls_back_on_a_failed_batch_and_sizes_the_run(tmp_path):
    """The whole timing cell with lerobot-train replaced: batch 32 fails, batch 16 logs 1.0 s steps."""
    calls = []

    class Sub:
        STDOUT = None

        @staticmethod
        def run(cmd, stdout, stderr):
            batch = int(next(a for a in cmd if a.startswith("--batch_size=")).split("=")[1])
            calls.append(batch)
            if batch == 32:
                stdout.write("torch.OutOfMemoryError: CUDA out of memory")
                return type("R", (), {"returncode": 1})()
            stdout.write("\n".join(f"step:{i} loss:0.1 updt_s:0.800 data_s:0.200" for i in range(20)))
            return type("R", (), {"returncode": 0})()

    ns = {"PY": "/venv/bin/python", "DS": "/ds", "TRAIN_HOURS": 2.5, "subprocess": Sub}
    exec(_train_cell().replace("/kaggle/working", str(tmp_path)).replace("/kaggle/temp", str(tmp_path)), ns)
    assert calls == [32, 16]
    assert (ns["BATCH"], ns["STEPS"]) == (16, 9000)          # 2.5 h at 1.0 s per step
    assert "updt_s" in (tmp_path / "timing.log").read_text()
