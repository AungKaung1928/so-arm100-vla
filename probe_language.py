"""Does the trained policy read the instruction (and look at the image)? Offline, on held-out demonstration frames.

    python probe_language.py --ckpt runs/policy_minilm_s0.pt --out runs/probe_language_minilm_s0.json

The validation episodes of the training run (same split as train.py) are scored with the
model's own masked L1 three times: with the recorded sentence, with the colour word
swapped for another cube's colour, and with a sentence for another task on the same
colour. The demonstrator went to the named cube, so a policy that uses the colour must
get worse when the colour is wrong. A last pass feeds the camera image of a window half the
set away (another episode, other cube positions). No simulator, no rendering: one process, one thread.
"""
import argparse

import numpy as np
import torch

from so_arm100_sim import instructions as I
from so_arm100_sim import scene

from so_arm100_vla.data import ChunkDataset, episode_split, load_flat
from so_arm100_vla.model import LangPolicy
from so_arm100_vla.policy import load_checkpoint, write_json
from train import to_torch

TEMPLATES = ("train", "heldout")


def task_colour(sentence):
    for task in I.TASKS:
        for colour in scene.CUBE_COLORS:
            if any(sentence in I.instructions(task, colour, t) for t in TEMPLATES):
                return task, colour
    raise KeyError(sentence)


def wrong_colour(sentence, rng):
    _, colour = task_colour(sentence)
    other = rng.choice([c for c in scene.CUBE_COLORS if c != colour])
    return sentence.replace(colour, other)


def wrong_task(sentence, rng):
    task, colour = task_colour(sentence)
    other = rng.choice([t for t in I.TASKS if t != task])
    return rng.choice(I.instructions(other, colour, "train"))


@torch.no_grad()
def l1(model, ds, encoder, swap=None, rng=None, batch_size=64):   # 64 = train.py's eval batch
    """Masked L1 over ds; `swap` maps a recorded sentence to the one fed instead."""
    cache = {}
    tot, n = 0.0, 0
    for b in ds.batches(batch_size, shuffle=False):
        img, st, lg, act, mask = to_torch(b)
        if swap is not None:
            fed = [cache.setdefault(s, swap(s, rng)) for s in b["sentence"]]
            lg = torch.as_tensor(np.stack(encoder.embed_all(fed)), dtype=torch.float32)
        tot += float(LangPolicy.loss(model(img, st, lg), act, mask)) * img.shape[0]
        n += img.shape[0]
    return tot / n


class SentenceDataset(ChunkDataset):
    """ChunkDataset that also hands back the recorded sentence of each window; with
    `image_shift`, the image comes from the window that far away (another episode)."""

    image_shift = 0

    def window(self, j):
        w = super().window(j)
        w["sentence"] = str(self.flat["tasks"][int(self.idx[j])])
        if self.image_shift:
            w["image"] = self.flat["images"][int(self.idx[(j + self.image_shift) % self.n])]
        return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/multitask")
    ap.add_argument("--repo-id", default="local/so_arm100_multitask")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, stats, encoder, ck = load_checkpoint(a.ckpt)
    flat = load_flat(a.root, a.repo_id)
    tr_eps, va_eps = episode_split(flat["episode"], seed=ck["cfg"]["seed"])
    res = {"ckpt": a.ckpt}
    for name, eps in (("val", va_eps), ("train", tr_eps)):
        ds = SentenceDataset(flat, eps, ck["cfg"]["k"], encoder, stats)
        rng = np.random.default_rng(0)
        res[name] = {"episodes": int(len(eps)), "windows": len(ds),
                     "l1_recorded": l1(model, ds, encoder),
                     "l1_wrong_colour": l1(model, ds, encoder, wrong_colour, rng),
                     "l1_wrong_task": l1(model, ds, encoder, wrong_task, rng)}
        ds.image_shift = len(ds) // 2
        res[name]["l1_other_image"] = l1(model, ds, encoder)
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()})
    if a.out:
        write_json(a.out, res)


if __name__ == "__main__":
    main()
