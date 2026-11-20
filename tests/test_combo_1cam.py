"""edge/combo_1cam.py summary: JSON values equal the printed table (p50/p99 ms, rounded) for a known input."""
import importlib.util, os

import numpy as np

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "edge", "combo_1cam.py")
_spec = importlib.util.spec_from_file_location("combo_1cam", _p)
combo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combo)  # no model load: heavy imports live in main()


def test_summarise_matches_hand_values():
    # 1.0 .. 2.0 s in 11 steps: median 1.5 s; p99 (linear) = 1.0 + 0.99 * 10 * 0.1 = 1.99 s
    t = list(np.linspace(1.0, 2.0, 11))
    s = combo.summarise({"pytorch fp32": t, "openvino fp32": [0.7] * 11}, {"pytorch fp32": 0.0, "openvino fp32": 2.1e-6}, 11)
    assert s["n"] == 11
    assert s["backends"]["pytorch fp32"] == {"p50_ms": 1500, "p99_ms": 1990, "max_abs_diff_vs_pytorch": 0.0}
    assert s["backends"]["openvino fp32"]["p50_ms"] == 700
    assert s["backends"]["openvino fp32"]["max_abs_diff_vs_pytorch"] == 2.1e-6


def test_summary_is_json_serialisable():
    import json
    s = combo.summarise({"a": [np.float64(0.0012)]}, {"a": np.float32(0.093)}, 1)
    assert json.loads(json.dumps(s))["backends"]["a"]["p50_ms"] == 1
