import numpy as np

from eval_floor import HoldRunner


def test_hold_runner_holds_the_first_pose_and_reset_forgets_it():
    r = HoldRunner()
    r.reset()
    a = {"state": np.arange(15.0)}
    b = {"state": np.arange(15.0) + 100.0}
    assert np.array_equal(r.act(a), np.arange(6.0))
    assert np.array_equal(r.act(b), np.arange(6.0))         # later steps keep the start pose
    r.reset()
    assert np.array_equal(r.act(b), np.arange(6.0) + 100.0)  # next episode holds its own start
