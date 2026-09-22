"""SmolVLA-base, zero-shot, in the same simulator. A reference, not a result.

    python smolvla_reference.py --dry-run            # mapping check, no download
    python smolvla_reference.py --run --episodes 10  # downloads ~1 GB, ~2 s per step

`lerobot/smolvla_base` was pretrained on community SO-100 / SO-101 datasets:
real arms, real cameras, joint positions in the units and normalisation of
those datasets. Running it here means feeding it a rendered 96x128 frame
resized to its 512x512 input, a 6-d joint vector in radians padded to its
state width, and the instruction string. Nothing is adapted. Expected
outcome: low success, and that is fine -- the point is to have the same
protocol, the same tasks and the same success criterion applied to a model
people know, next to the small policy trained here. It is NOT a comparison
of capability; it is the calibration line that says what "zero-shot from a
different embodiment" is worth on this bench.

Mapping (built from the policy's own `config.input_features` at runtime):

    observation.images.<name>   every image feature the config lists gets the
                                front camera, resized with padding to the
                                config's `resize_imgs_with_padding`
    observation.state           the 6 joint angles (radians), zero-padded to
                                the feature's length
    task                        the instruction string

Caveats, all stated in the README: different embodiment normalisation
statistics, different camera placement, radians against a model trained on
motor units, and a 20 Hz control loop against 30 fps datasets.
"""
import argparse
import time

import numpy as np

from so_arm100_sim.evaluate import evaluate, provenance
from so_arm100_sim import instructions as I

from so_arm100_vla import splits as S
from so_arm100_vla.data import IMAGE
from so_arm100_vla.policy import write_json

REPO = "lerobot/smolvla_base"


def resize_pad(img_uint8_hwc, size):
    """Letterbox an HxWx3 uint8 image into size x size, float CHW in [0,1]."""
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(np.asarray(img_uint8_hwc)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    h, w = x.shape[-2:]
    s = min(size / h, size / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    out = torch.zeros(1, 3, size, size)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[:, :, top:top + nh, left:left + nw] = x
    return out


def build_batch(obs, instruction, input_features, resize):
    """Bench observation -> the dict `select_action` expects."""
    import torch
    batch = {"task": [instruction]}
    q = np.asarray(obs["state"][:6], np.float32)
    for key, feat in input_features.items():
        shape = tuple(getattr(feat, "shape", feat if isinstance(feat, (tuple, list)) else ()))
        if key.startswith("observation.image"):
            batch[key] = resize_pad(obs["image"], resize[0] if resize else 224)
        elif key == "observation.state":
            n = int(shape[0]) if shape else 6
            v = np.zeros(n, np.float32)
            v[:min(6, n)] = q[:min(6, n)]
            batch[key] = torch.as_tensor(v).unsqueeze(0)
    return batch


class StubPolicy:
    """Stands in for SmolVLAPolicy in --dry-run and in the tests."""

    class Cfg:
        def __init__(self):
            self.input_features = {"observation.images.camera1": (3, 512, 512),
                                   "observation.images.camera2": (3, 512, 512),
                                   "observation.state": (6,)}
            self.resize_imgs_with_padding = (512, 512)

    def __init__(self):
        self.config = self.Cfg()
        self.calls = []

    def reset(self):
        pass

    def select_action(self, batch):
        import torch
        self.calls.append(sorted(batch))
        for k in self.config.input_features:
            assert k in batch, f"missing {k}"
        assert batch["observation.images.camera1"].shape == (1, 3, 512, 512)
        assert batch["observation.state"].shape == (1, 6)
        assert isinstance(batch["task"], list) and isinstance(batch["task"][0], str)
        return torch.zeros(1, 6)


class SmolVLARunner:
    def __init__(self, policy, sentences, seed=0, home=None, log_time=None):
        self.policy = policy
        self.sentences = list(sentences)
        self.rng = np.random.default_rng(seed)
        self.resize = getattr(policy.config, "resize_imgs_with_padding", (512, 512))
        self.features = policy.config.input_features
        self.times = log_time if log_time is not None else []
        self.home = home

    def reset(self):
        self.sentence = self.sentences[int(self.rng.integers(len(self.sentences)))]
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, obs):
        import torch
        batch = build_batch(obs, self.sentence, self.features, self.resize)
        t0 = time.perf_counter()
        with torch.no_grad():
            a = self.policy.select_action(batch)
        self.times.append(time.perf_counter() - t0)
        a = np.asarray(a).reshape(-1)[:6].astype(np.float64)
        # The model emits targets in its training units; absolute radians here.
        # A zero output holds the home pose, which is the only safe default.
        return a if np.isfinite(a).all() else (self.home if self.home is not None else np.zeros(6))


def load_policy():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(REPO)
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stub policy, no download")
    ap.add_argument("--run", action="store_true", help="download and run SmolVLA-base")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tasks", nargs="*", default=list(I.TASKS))
    ap.add_argument("--color", default="red")
    ap.add_argument("--out", default="runs/smolvla_zero_shot.json")
    a = ap.parse_args()
    if not (a.dry_run or a.run):
        raise SystemExit("pass --dry-run (no download) or --run (downloads ~1 GB)")
    policy = StubPolicy() if a.dry_run else load_policy()
    times, cells = [], {}
    for task in a.tasks:
        sents = S.sentences_for("seen", task, a.color) if (task, a.color) in I.TRAIN_COMBOS \
            else I.instructions(task, a.color, "train")
        r = evaluate(lambda env, _s=sents: SmolVLARunner(policy, _s, log_time=times),
                     f"{task}:{a.color}:distractors", n_episodes=a.episodes,
                     seeds=tuple(range(a.seeds)),
                     env_kwargs={"obs_mode": "proprio", "action_mode": "absolute",
                                 "image": dict(IMAGE)})
        cells[task] = r
        print(f"  {task:11s} success {r['success_rate']:.2f}  ({'stub' if a.dry_run else REPO})")
    out = {"model": "stub" if a.dry_run else REPO, "episodes": a.episodes, "cells": cells,
           "forward_s_p50": float(np.percentile(times, 50)) if times else None,
           "forward_s_p90": float(np.percentile(times, 90)) if times else None,
           "n_forward": len(times), **provenance()}
    write_json(a.out if a.run else a.out.replace(".json", "_dryrun.json"), out)
    if times:
        print(f"forward pass p50 {out['forward_s_p50']:.3f} s, p90 {out['forward_s_p90']:.3f} s "
              f"over {len(times)} calls")


if __name__ == "__main__":
    main()
