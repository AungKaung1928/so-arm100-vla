"""LangRunner, the wrapper every evaluation runs through, with a stub network.

The network is replaced by a function that records its inputs and returns a fixed
chunk, so the wrapper's own arithmetic is checked exactly: state normalised on
the way in, chunk denormalised on the way out, ensembler cleared per episode, and
a sentence drawn from the pool at every reset.
"""
import numpy as np
import torch

from so_arm100_sim.env import ACT_DIM
from so_arm100_vla import text as T
from so_arm100_vla.data import Stats
from so_arm100_vla.policy import LangRunner

K = 3


class StubNet:
    k = K

    def __init__(self):
        self.states, self.chunk = [], torch.zeros(1, K, ACT_DIM)

    def __call__(self, img, state, lang):
        self.states.append(state.numpy().copy())
        return self.chunk.clone()


def _runner(sentences=("pick up the red cube",), seed=0):
    stats = Stats(np.full(15, 1.0), np.full(15, 2.0), np.full(ACT_DIM, 3.0), np.full(ACT_DIM, 4.0))
    net = StubNet()
    return LangRunner(net, stats, T.HashEncoder(), list(sentences), seed=seed, m=0.0), net


def _obs(v=5.0):
    return {"image": np.zeros((8, 8, 3), np.uint8), "state": np.full(15, v, np.float32)}


def test_state_in_normalised_action_out_denormalised():
    r, net = _runner()
    net.chunk[:] = 0.5
    a = r.act(_obs(5.0))
    assert np.allclose(net.states[0], (5.0 - 1.0) / 2.0)          # norm_state
    assert np.allclose(a, 0.5 * 4.0 + 3.0)                         # denorm_action


def test_reset_clears_the_ensembler():
    r, net = _runner()
    r.reset()
    net.chunk[:] = 1.0
    r.act(_obs())
    r.reset()                                                      # new episode
    net.chunk[:] = -1.0
    # m=0 averages live chunks: a leftover chunk from the last episode would give 3 + 4*0
    assert np.allclose(r.act(_obs()), -1.0 * 4.0 + 3.0)


def test_reset_draws_from_the_whole_pool():
    pool = [f"pick up the {c} cube" for c in ("red", "green", "blue")]
    r, _ = _runner(pool, seed=1)
    seen = set()
    for _ in range(30):
        r.reset()
        seen.add(r.sentence)
    assert seen == set(pool)
