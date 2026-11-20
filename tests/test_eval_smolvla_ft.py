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
