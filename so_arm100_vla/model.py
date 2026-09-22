"""The policy: three tokens in, a chunk of joint targets out.

    image   -> small CNN -> (FiLM by language) -> 32 spatial soft-argmax
               keypoints -> 64 numbers -> token
    proprio -> linear -> token
    language embedding (384-d, frozen encoder) -> linear -> token

    transformer encoder over the three tokens (d=128, 2 layers)
    k learned action queries -> transformer decoder -> k x 6 joint targets

Why keypoints rather than a flattened feature map: a policy has to say WHERE
the cube is, and a spatial soft-argmax hands position to the network instead
of making it learn coordinates from scratch (the same front end the
pose-regression project measured as the better head, at 5x fewer
parameters). Why FiLM: the instruction should change what the vision front
end looks for -- "the blue cube" should sharpen blue -- and FiLM is the
cheapest way to let language modulate features before they are pooled.
Why chunks: one forward pass per k steps of intent, temporally ensembled at
inference, is what made ACT work on real arms with a few dozen demos.

L1 loss on normalised actions, masked past the episode's end.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from so_arm100_sim.env import PROPRIO_DIM, ACT_DIM

from .text import EMBED_DIM


def conv_bn(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SpatialSoftArgmax(nn.Module):
    """(B,K,H,W) -> (B,2K) expected (u,v) in [-1,1] per channel."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, x):
        b, k, h, w = x.shape
        p = F.softmax(x.reshape(b, k, h * w) / self.log_t.exp(), dim=-1).reshape(b, k, h, w)
        u = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        v = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        return torch.cat([(p * u).sum((2, 3)), (p * v).sum((2, 3))], dim=1)


class LangPolicy(nn.Module):
    def __init__(self, k=10, d=128, n_kp=32, layers=2, heads=4, film=True,
                 embed_dim=EMBED_DIM, state_dim=PROPRIO_DIM, act_dim=ACT_DIM, feat=64):
        super().__init__()
        self.k, self.d, self.film = int(k), int(d), bool(film)
        self.trunk = nn.Sequential(conv_bn(3, 16, 2), conv_bn(16, 32, 2), conv_bn(32, feat, 2))
        if self.film:
            self.film_gen = nn.Linear(embed_dim, 2 * feat)
        self.kp = nn.Conv2d(feat, n_kp, 1)
        self.ssa = SpatialSoftArgmax()
        self.img_token = nn.Linear(2 * n_kp, d)
        self.state_token = nn.Linear(state_dim, d)
        self.lang_token = nn.Linear(embed_dim, d)
        self.type_embed = nn.Parameter(torch.zeros(3, d))
        enc_layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        dec_layer = nn.TransformerDecoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, layers)
        self.queries = nn.Parameter(torch.randn(self.k, d) * 0.02)
        self.head = nn.Linear(d, act_dim)
        self.register_buffer("img_mean", torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode_image(self, image, lang):
        """image: (B,3,H,W) float in [0,1]."""
        z = self.trunk((image - self.img_mean) / self.img_std)
        if self.film:
            gb = self.film_gen(lang)
            gamma, beta = gb.chunk(2, dim=-1)
            z = z * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.ssa(self.kp(z))

    def forward(self, image, state, lang):
        """-> (B, k, act_dim) normalised actions."""
        kp = self.encode_image(image, lang)
        tokens = torch.stack([self.lang_token(lang), self.img_token(kp),
                              self.state_token(state)], dim=1) + self.type_embed
        mem = self.encoder(tokens)
        q = self.queries.unsqueeze(0).expand(image.shape[0], -1, -1)
        return self.head(self.decoder(q, mem))

    @staticmethod
    def loss(pred, target, mask):
        l1 = (pred - target).abs().mean(-1)          # (B,k)
        return (l1 * mask).sum() / mask.sum().clamp(min=1.0)


def image_to_tensor(img_uint8_hwc):
    """(B,H,W,3) or (H,W,3) uint8 -> (B,3,H,W) float in [0,1]."""
    x = torch.as_tensor(np.asarray(img_uint8_hwc))
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.permute(0, 3, 1, 2).float() / 255.0


class ChunkEnsembler:
    """ACT-style temporal ensembling.

    Every step the policy predicts k actions for t..t+k-1. The action executed
    at t is the exponentially weighted average of all still-live predictions
    for t, older predictions weighted by exp(-m * age). m=0.01 is ACT's value;
    m -> inf recovers "use only the newest chunk".
    """

    def __init__(self, k, act_dim, m=0.01):
        self.k, self.act_dim, self.m = int(k), int(act_dim), float(m)
        self.reset()

    def reset(self):
        self.t = 0
        self.buffer = []          # (t_issued, chunk (k, act_dim))

    def push(self, chunk):
        self.buffer.append((self.t, np.asarray(chunk, np.float64)))
        self.buffer = [(ti, c) for ti, c in self.buffer if ti + self.k > self.t]

    def action(self):
        acts, ws = [], []
        for ti, c in self.buffer:
            age = self.t - ti
            if 0 <= age < self.k:
                acts.append(c[age])
                ws.append(np.exp(-self.m * age))
        self.t += 1
        if not acts:
            raise RuntimeError("no live chunk; call push() before action()")
        w = np.asarray(ws)
        return (np.stack(acts) * w[:, None]).sum(0) / w.sum()

    @staticmethod
    def weights(n, m=0.01):
        """The normalised weights n live predictions of ages 0..n-1 would get."""
        w = np.exp(-m * np.arange(n))
        return w / w.sum()
