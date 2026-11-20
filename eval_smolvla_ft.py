"""Evaluate a fine-tuned SmolVLA checkpoint (from kaggle/smolvla_finetune.ipynb) on the bench.

    python eval_smolvla_ft.py --ckpt runs/smolvla_sim/pretrained_model --episodes 10

Same protocol as eval_language.py (`task:colour:distractors`, seen / paraphrase /
combo splits). Unlike smolvla_reference.py, observations go through the
checkpoint's own LeRobot pre/post processors, so state and actions are
normalized with the fine-tune dataset's stats. Inputs match `record --smolvla`:
6-d state and actions in SO-100 units, cameras top/front/side as camera1..3.

CPU inference is about 3 s per 50-step chunk (edge/ in the README has the measured p50).
"""
import argparse
import time

import numpy as np

from so_arm100_sim import instructions as I
from so_arm100_sim import smolvla_export as SX
from so_arm100_sim.evaluate import evaluate, provenance
from so_arm100_vla import splits as S
from so_arm100_vla.data import IMAGE
from so_arm100_vla.policy import write_json


def observation(env, obs, task):
    import torch
    h, w = IMAGE["height"], IMAGE["width"]
    out = {"observation.state": torch.from_numpy(SX.to_so100_units(obs["state"][:6]))[None],
           "task": [task]}
    for key, cam in SX.CAMERA_MAP.items():
        img = env.render(cam, h, w)
        out[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
    return out


class FineTunedRunner:
    def __init__(self, env, policy, pre, post, sentences, seed=0, times=None):
        self.env, self.policy, self.pre, self.post = env, policy, pre, post
        self.sentences = list(sentences)
        self.rng = np.random.default_rng(seed)
        self.times = times if times is not None else []

    def reset(self):
        self.sentence = self.sentences[int(self.rng.integers(len(self.sentences)))]
        self.policy.reset()

    def act(self, obs):
        import torch
        computes = len(self.policy._queues.get("action", [])) == 0
        t0 = time.perf_counter()
        with torch.no_grad():
            a = self.post(self.policy.select_action(self.pre(observation(self.env, obs, self.sentence))))
        if computes:
            self.times.append(time.perf_counter() - t0)
        return SX.from_so100_units(np.asarray(a).reshape(-1)[:6])


def load(ckpt):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.config.device = "cpu"
    policy.to("cpu").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}})
    return policy, pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir of the fine-tune")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--splits", nargs="*", default=["seen", "paraphrase", "combo"])
    ap.add_argument("--out", default="runs/smolvla_finetuned.json")
    a = ap.parse_args()
    policy, pre, post = load(a.ckpt)
    times, cells = [], {}
    for split in a.splits:
        pairs = I.HELDOUT_COMBOS if split == "combo" else I.TRAIN_COMBOS
        for task, color in pairs:
            sents = S.sentences_for(split, task, color)
            r = evaluate(lambda env, _s=sents: FineTunedRunner(env, policy, pre, post, _s, times=times),
                         f"{task}:{color}:distractors", n_episodes=a.episodes,
                         seeds=tuple(range(a.seeds)),
                         env_kwargs={"obs_mode": "proprio", "action_mode": "absolute",
                                     "image": dict(IMAGE)})
            cells[f"{split}/{task}:{color}"] = r
            print(f"  {split:10s} {task}:{color:6s} success {r['success_rate']:.2f}", flush=True)
    out = {"model": a.ckpt, "episodes": a.episodes, "seeds": a.seeds, "cells": cells,
           "forward_s_p50": float(np.percentile(times, 50)) if times else None,
           "n_forward": len(times), **provenance()}
    write_json(a.out, out)


if __name__ == "__main__":
    main()
