"""Running a trained LangPolicy inside the bench's evaluation protocol.

The bench environment knows nothing about language, so the instruction lives
here: `LangRunner` is built with a sentence (or a pool of sentences to draw
one from at every `reset()`), embeds it once, and closes over it for the
episode. `evaluate(policy_factory, ...)` then works unchanged.
"""
import json
import os

import numpy as np
import torch

from so_arm100_sim.env import ACT_DIM

from . import text as T
from .data import Stats
from .model import LangPolicy, ChunkEnsembler, image_to_tensor


def save_checkpoint(path, model, stats, encoder_name, cfg, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "stats": stats.to_dict(),
                "encoder": encoder_name, "cfg": cfg, "extra": extra or {}}, path)


def load_checkpoint(path, encoder=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = LangPolicy(k=cfg["k"], d=cfg["d"], n_kp=cfg["n_kp"], layers=cfg["layers"],
                       heads=cfg["heads"], film=cfg["film"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    stats = Stats.from_dict(ck["stats"])
    if encoder is None:
        encoder = T.make_encoder(ck["encoder"])
    return model, stats, encoder, ck


class LangRunner:
    """policy_factory-compatible: `.reset()` picks the sentence, `.act(obs)` acts."""

    def __init__(self, model, stats, encoder, sentences, seed=0, m=0.01, log_sentences=None):
        self.model, self.stats, self.encoder = model, stats, encoder
        self.sentences = [sentences] if isinstance(sentences, str) else list(sentences)
        self.rng = np.random.default_rng(seed)
        self.ens = ChunkEnsembler(model.k, ACT_DIM, m=m)
        self.log = log_sentences
        self._embeds = {s: v for s, v in zip(self.sentences, encoder.embed_all(self.sentences))}
        self.sentence = None
        self.n_forward = 0

    def reset(self):
        self.sentence = self.sentences[int(self.rng.integers(len(self.sentences)))]
        self._lang = torch.as_tensor(self._embeds[self.sentence]).unsqueeze(0)
        self.ens.reset()
        if self.log is not None:
            self.log.append(self.sentence)

    @torch.no_grad()
    def act(self, obs):
        if self.sentence is None:
            self.reset()
        img = image_to_tensor(obs["image"])
        state = torch.as_tensor(self.stats.norm_state(obs["state"]), dtype=torch.float32).unsqueeze(0)
        chunk = self.model(img, state, self._lang)[0].numpy()
        self.n_forward += 1
        self.ens.push(self.stats.denorm_action(chunk))
        return self.ens.action()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
