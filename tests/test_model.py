import numpy as np
import torch

from so_arm100_vla.model import LangPolicy, ChunkEnsembler, SpatialSoftArgmax, image_to_tensor


def test_forward_shapes_with_and_without_film():
    for film in (True, False):
        m = LangPolicy(k=10, d=64, n_kp=8, layers=1, heads=4, film=film, feat=16)
        img = torch.rand(3, 3, 48, 64)
        out = m(img, torch.zeros(3, 15), torch.zeros(3, 384))
        assert out.shape == (3, 10, 6)
        assert m.n_params() > 0
    # FiLM adds parameters, and only FiLM
    a = LangPolicy(k=10, d=64, n_kp=8, layers=1, heads=4, film=True, feat=16).n_params()
    b = LangPolicy(k=10, d=64, n_kp=8, layers=1, heads=4, film=False, feat=16).n_params()
    assert a - b == 384 * 32 + 32



def test_soft_argmax_pixel_convention():
    # Saved checkpoints learned their heads on this convention: u = column, v = row, -1 at
    # index 0 and +1 at the last index. A flip or swap still trains, but breaks every checkpoint.
    h, w = 5, 7
    x = torch.full((1, 2, h, w), -1e4)
    x[0, 0, 1, 6] = 0.0          # row 1, last column
    x[0, 1, 4, 0] = 0.0          # last row, column 0
    out = SpatialSoftArgmax()(x)[0]
    torch.testing.assert_close(out, torch.tensor([1.0, -1.0, -0.5, 1.0]))   # (u0, u1, v0, v1)
    # uniform map -> centre
    torch.testing.assert_close(SpatialSoftArgmax()(torch.zeros(1, 1, h, w)), torch.zeros(1, 2))


def test_language_changes_the_output():
    torch.manual_seed(0)
    m = LangPolicy(k=5, d=64, n_kp=8, layers=1, heads=4, feat=16).eval()
    img, st = torch.rand(1, 3, 32, 48), torch.zeros(1, 15)
    a = m(img, st, torch.randn(1, 384))
    b = m(img, st, torch.randn(1, 384))
    assert not torch.allclose(a, b)


def test_image_sizes_are_free():
    m = LangPolicy(k=4, d=64, n_kp=8, layers=1, heads=4, feat=16).eval()
    for h, w in ((32, 48), (96, 128)):
        assert m(torch.rand(1, 3, h, w), torch.zeros(1, 15), torch.zeros(1, 384)).shape == (1, 4, 6)


def test_masked_l1_loss():
    pred = torch.zeros(2, 3, 6)
    tgt = torch.ones(2, 3, 6)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    assert float(LangPolicy.loss(pred, tgt, mask)) == 1.0
    tgt[:, 2] = 100.0            # masked steps must not count
    assert float(LangPolicy.loss(pred, tgt, mask)) == 1.0


def test_image_to_tensor():
    x = image_to_tensor(np.full((4, 6, 3), 255, np.uint8))
    assert x.shape == (1, 3, 4, 6) and float(x.max()) == 1.0


def test_chunk_ensembler_weights_and_averaging():
    w = ChunkEnsembler.weights(3, m=0.01)
    assert abs(w.sum() - 1) < 1e-12 and w[0] > w[1] > w[2]
    ens = ChunkEnsembler(k=3, act_dim=1, m=0.0)      # m=0: plain average
    ens.push(np.array([[1.0], [2.0], [3.0]]))
    assert ens.action()[0] == 1.0                     # t=0: only chunk 0, age 0
    ens.push(np.array([[10.0], [20.0], [30.0]]))      # issued at t=1
    # t=1: chunk0 age1 -> 2.0, chunk1 age0 -> 10.0, equal weights
    assert abs(ens.action()[0] - 6.0) < 1e-12
    ens.push(np.array([[100.0], [200.0], [300.0]]))   # t=2
    # t=2: chunk0 age2 -> 3, chunk1 age1 -> 20, chunk2 age0 -> 100
    assert abs(ens.action()[0] - 41.0) < 1e-12
    ens.push(np.array([[0.0], [0.0], [0.0]]))          # t=3: chunk0 expired
    assert abs(ens.action()[0] - (30 + 200 + 0) / 3) < 1e-12


def test_chunk_ensembler_action_weights_newest_most():
    # the m=0 case above cannot see the weighting direction; with m=1 the newest
    # chunk (age 0) must get weight 1 and the older one exp(-1), as weights() states
    ens = ChunkEnsembler(k=3, act_dim=1, m=1.0)
    ens.push(np.array([[0.0], [2.0], [0.0]]))          # t=0
    ens.action()
    ens.push(np.array([[10.0], [0.0], [0.0]]))         # t=1: old chunk age 1 -> 2, new age 0 -> 10
    w = ChunkEnsembler.weights(2, m=1.0)               # [age 0, age 1]
    assert abs(ens.action()[0] - (w[0] * 10.0 + w[1] * 2.0)) < 1e-12
    assert w[0] > w[1]
