"""The Kaggle notebook is generated; check it builds, parses, and records only training pairs."""
import ast
import importlib.util
import os

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
    combos = [tuple(s.split(":")) for s in mk.COMBOS.split()]
    assert combos == list(I.TRAIN_COMBOS)
    assert not set(combos) & set(I.HELDOUT_COMBOS)


def test_s_per_step_skips_warmup():
    ns = {}
    src = next("".join(c["source"]) for c in mk.build()["cells"]
               if "def s_per_step" in "".join(c["source"]))
    exec(src.split("for BATCH")[0], ns)   # defs only, no training
    log = "updt_s:80.0 x updt_s:9.0 y updt_s:1.0 z updt_s:3.0"
    assert ns["s_per_step"](log) == pytest.approx(2.0)
