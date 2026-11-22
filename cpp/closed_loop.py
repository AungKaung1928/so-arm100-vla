"""Closed loop, PyTorch vs the exported ONNX graph, in lockstep on two identically seeded envs.

Open-loop parity (cpp/bench.py) says one step agrees to ~1e-6 rad. In closed loop each
policy sees the state its own actions produced, so a small difference can grow. Both runners
are LangRunner (same sentence draw, same ensembler); only the network call differs. Needs
cpp/bench.py's work dir (policy.onnx). One torch and one ORT thread; rendering adds its own
(8 episodes: 64 s wall).

    python cpp/closed_loop.py --episodes 8 --out runs/act_closed_loop.json
    python cpp/closed_loop.py --episodes 8 --control --out runs/act_closed_loop_control.json
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from so_arm100_sim import instructions as I
from so_arm100_sim.env import ArmEnv
from so_arm100_vla.data import IMAGE
from so_arm100_vla.policy import LangRunner, load_checkpoint


class OrtModel:
    """Stands in for LangPolicy inside LangRunner: same call, ONNX Runtime underneath."""

    def __init__(self, path, k):
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.k = k

    def __call__(self, image, state, lang):
        out = self.sess.run(None, {"image": image.numpy(), "state": state.numpy(), "lang": lang.numpy()})[0]
        return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/policy_hash_s0.pt")
    ap.add_argument("--onnx", default="out/act_cpp/policy.onnx", help="cpp/bench.py export of --ckpt")
    ap.add_argument("--task", default="reach:red")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="both runners PyTorch: the divergence the simulator alone produces")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, stats, encoder, _ = load_checkpoint(a.ckpt)
    task, colour = a.task.split(":")
    sentences = I.instructions(task, colour, "train")
    kw = {"obs_mode": "proprio", "action_mode": "absolute", "image": dict(IMAGE)}
    runners = [LangRunner(model, stats, encoder, sentences, seed=a.seed),
               LangRunner(model if a.control else OrtModel(a.onnx, model.k), stats, encoder,
                          sentences, seed=a.seed)]
    eps, t0 = [], time.perf_counter()
    for j in range(a.episodes):
        envs = [ArmEnv(f"{task}:{colour}:distractors", seed=a.seed, **kw) for _ in range(2)]
        ep_seed = a.seed * 10_007 + j                     # so_arm100_sim.evaluate's episode seeds
        obs = [e.reset(seed=ep_seed) for e in envs]
        for r in runners:
            r.reset()
        assert runners[0].sentence == runners[1].sentence
        assert np.array_equal(obs[0]["image"], obs[1]["image"]), "envs must start identical"
        diffs, done, success = [], [False, False], [False, False]
        while not all(done):
            acts = [None if done[i] else runners[i].act(obs[i]) for i in range(2)]
            if not any(done):
                diffs.append(float(np.abs(acts[0] - acts[1]).max()))
            for i in range(2):
                if not done[i]:
                    obs[i], _, done[i], info = envs[i].step(acts[i])
                    success[i] = success[i] or bool(info["success_ever"])
        for e in envs:
            e.close()
        d = np.asarray(diffs)
        eps.append({"seed": ep_seed, "steps_lockstep": len(d), "success_torch": success[0],
                    "success_onnx": success[1], "max_action_diff": float(d.max()),
                    "first_step_over_1e-3": int(np.argmax(d > 1e-3)) if (d > 1e-3).any() else None})
        print(eps[-1], flush=True)
    res = {"ckpt": a.ckpt, "second": "pytorch (control)" if a.control else a.onnx, "task": a.task, "episodes": a.episodes, "threads": 1,
           "onnxruntime": ort.__version__, "wall_s": time.perf_counter() - t0,
           "success_torch": sum(e["success_torch"] for e in eps),
           "success_onnx": sum(e["success_onnx"] for e in eps),
           "same_outcome": sum(e["success_torch"] == e["success_onnx"] for e in eps),
           "max_action_diff": max(e["max_action_diff"] for e in eps), "per_episode": eps}
    print(json.dumps({k: v for k, v in res.items() if k != "per_episode"}, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
