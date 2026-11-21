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
    """Masked L1 over ds (train.py's batch weighting) and, per episode, the summed masked
    L1 and mask, for the bootstrap. `swap` maps a recorded sentence to the one fed instead."""
    cache = {}
    tot, n = 0.0, 0
    num, den = {}, {}
    for b in ds.batches(batch_size, shuffle=False):
        img, st, lg, act, mask = to_torch(b)
        if swap is not None:
            fed = [cache.setdefault(s, swap(s, rng)) for s in b["sentence"]]
            lg = torch.as_tensor(np.stack(encoder.embed_all(fed)), dtype=torch.float32)
        pred = model(img, st, lg)
        tot += float(LangPolicy.loss(pred, act, mask)) * img.shape[0]
        n += img.shape[0]
        per = ((pred - act).abs().mean(-1) * mask).sum(-1).numpy()
        for e, x, m in zip(b["episode"], per, mask.sum(-1).numpy()):
            num[int(e)] = num.get(int(e), 0.0) + float(x)
            den[int(e)] = den.get(int(e), 0.0) + float(m)
    eps = sorted(num)
    return tot / n, np.array([num[e] for e in eps]), np.array([den[e] for e in eps])


def change_ci(base, cond, n_boot=2000, seed=0):
    """Relative L1 change cond/base - 1 with a 95 % bootstrap interval over episodes (paired)."""
    _, bn, bd = base
    _, cn, cd = cond
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bn), size=(n_boot, len(bn)))
    boot = (cn[idx].sum(1) / cd[idx].sum(1)) / (bn[idx].sum(1) / bd[idx].sum(1)) - 1
    return [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]


class SentenceDataset(ChunkDataset):
    """ChunkDataset that also hands back the recorded sentence of each window; with
    `image_shift`, the image comes from the window that far away (another episode)."""

    image_shift = 0

    def window(self, j):
        w = super().window(j)
        w["sentence"] = str(self.flat["tasks"][int(self.idx[j])])
        w["episode"] = int(self.flat["episode"][int(self.idx[j])])
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
        runs = {"recorded": l1(model, ds, encoder),
                "wrong_colour": l1(model, ds, encoder, wrong_colour, rng),
                "wrong_task": l1(model, ds, encoder, wrong_task, rng)}
        ds.image_shift = len(ds) // 2
        runs["other_image"] = l1(model, ds, encoder)
        res[name] = {"episodes": int(len(eps)), "windows": len(ds),
                     **{f"l1_{k}": v[0] for k, v in runs.items()},
                     **{f"ci95_change_{k}": change_ci(runs["recorded"], v)
                        for k, v in runs.items() if k != "recorded"}}
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()})
    if a.out:
        write_json(a.out, res)


if __name__ == "__main__":
    main()
