"""eval_smolvla_ft input/output wiring with a stub policy (no checkpoint, no download)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
E = pytest.importorskip("eval_smolvla_ft")
SX = E.SX


class FakeEnv:
    def render(self, cam, h, w):
        return np.full((h, w, 3), {"top": 10, "front": 20, "side": 30}[cam], np.uint8)


class StubPolicy:
    def __init__(self):
        self._queues = {"action": []}
        self.seen = None

    def reset(self):
        pass

    def select_action(self, batch):
        self.seen = batch
        return torch.tensor([[90.0, -90.0, 0.0, 45.0, 0.0, 100.0]])


def test_observation_layout_and_units():
    obs = {"state": np.concatenate([[np.pi / 2, 0, 0, 0, 0, SX.JAW_RANGE[0]], np.zeros(9)])}
    b = E.observation(FakeEnv(), obs, "lift the red cube")
    h, w = E.IMAGE["height"], E.IMAGE["width"]
    assert b["observation.state"].shape == (1, 6)
    assert b["observation.state"][0, 0].item() == pytest.approx(90.0, abs=1e-3)
    assert b["observation.state"][0, 5].item() == pytest.approx(0.0, abs=1e-3)
    assert b["observation.images.camera1"].shape == (1, 3, h, w)
    assert b["observation.images.camera1"].max().item() == pytest.approx(10 / 255)   # top
    assert b["observation.images.camera3"].max().item() == pytest.approx(30 / 255)   # side
    assert b["task"] == ["lift the red cube"]


def test_action_goes_back_to_radians():
    p = StubPolicy()
    r = E.FineTunedRunner(FakeEnv(), p, lambda x: x, lambda a: a, ["reach the red cube"])
    r.reset()
    q = r.act({"state": np.zeros(15)})
    np.testing.assert_allclose(q[:5], np.radians([90.0, -90.0, 0.0, 45.0, 0.0]), atol=1e-6)
    assert q[5] == pytest.approx(SX.JAW_RANGE[1])
    assert len(r.times) == 1 and "observation.images.camera2" in p.seen


def test_main_closed_loop_with_stub_policy(tmp_path, monkeypatch):
    """main() end to end on the real env (3 cameras rendered per step) with the stub in place of
    the checkpoint: wiring and JSON only, no success rate is meaningful here."""
    import json
    import sys

    from so_arm100_sim import render
    if not render.available():
        pytest.skip("no offscreen GL context")
    p = StubPolicy()
    monkeypatch.setattr(E, "load", lambda ckpt: (p, lambda x: x, lambda a: a))
    monkeypatch.setattr(E.I, "HELDOUT_COMBOS", [("lift", "blue")])   # one cell keeps it short
    out = tmp_path / "ft.json"
    monkeypatch.setattr(sys, "argv", ["eval_smolvla_ft.py", "--ckpt", "stub", "--episodes", "1",
                                      "--splits", "combo", "--out", str(out)])
    E.main()
    r = json.loads(out.read_text())
    assert list(r["cells"]) == ["combo/lift:blue"]
    assert all(0.0 <= c["success_rate"] <= 1.0 for c in r["cells"].values())
    assert r["n_forward"] == 150                 # one forward per control step of a lift episode
    assert sorted(p.seen) == ["observation.images.camera1", "observation.images.camera2",
                              "observation.images.camera3", "observation.state", "task"]
