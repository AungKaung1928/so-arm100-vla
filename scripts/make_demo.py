"""docs/media/demo.gif: the tracked hash-encoder policy on reach:red, one episode it solves
and one it does not, side by side at real time (20 Hz control, 20 fps).

Same runner, seeds and episode order as cpp/closed_loop.py, so each outcome must equal the
PyTorch outcome in runs/act_closed_loop.json; the script stops if it does not. One torch
thread; about 40 s on a laptop CPU.

    MUJOCO_GL=glfw python scripts/make_demo.py
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from so_arm100_sim import instructions as I
from so_arm100_sim.env import ArmEnv
from so_arm100_sim.render import Renderer
from so_arm100_vla.data import IMAGE
from so_arm100_vla.policy import LangRunner, load_checkpoint

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT, TASK, COLOUR, SEED = "runs/policy_hash_s0.pt", "reach", "red", 0
SHOW = (4, 0)                  # episode indices: one success, one failure in the tracked JSON
H, W, BAR = 240, 320, 28


def main():
    torch.set_num_threads(1)
    ref = {e["seed"]: e["success_torch"]
           for e in json.load(open(os.path.join(ROOT, "runs/act_closed_loop.json")))["per_episode"]}
    model, stats, encoder, _ = load_checkpoint(os.path.join(ROOT, CKPT))
    runner = LangRunner(model, stats, encoder, I.instructions(TASK, COLOUR, "train"), seed=SEED)
    kw = {"obs_mode": "proprio", "action_mode": "absolute", "image": dict(IMAGE)}
    clips = {}
    for j in range(max(SHOW) + 1):        # every episode runs, so the sentence draws match closed_loop.py
        env = ArmEnv(f"{TASK}:{COLOUR}:distractors", seed=SEED, **kw)
        ep_seed = SEED * 10_007 + j
        obs = env.reset(seed=ep_seed)
        runner.reset()
        cam = Renderer(env.model, H, W) if j in SHOW else None
        frames, done, success = [], False, False
        while not done:
            if cam:
                frames.append(cam.render(env.data, "front"))
            obs, _, done, info = env.step(runner.act(obs))
            success = success or bool(info["success_ever"])
        if cam:
            frames.append(cam.render(env.data, "front"))
            cam.close()
            clips[j] = (frames, success, runner.sentence)
        env.close()
        assert success == ref[ep_seed], f"episode {j}: {success} here, {ref[ep_seed]} in the JSON"
    n = max(len(c[0]) for c in clips.values())
    out = []
    for t in range(n):
        canvas = Image.new("RGB", (W * len(SHOW), H + BAR), "white")
        d = ImageDraw.Draw(canvas)
        for k, j in enumerate(SHOW):
            frames, success, sentence = clips[j]
            canvas.paste(Image.fromarray(frames[min(t, len(frames) - 1)]), (k * W, BAR))
            end = len(frames) - 1
            label = f'"{sentence}"  ' + (f"step {t}" if t < end else
                                         f"success at step {end}" if success else f"no success in {end} steps")
            d.text((k * W + 6, 8), label, fill="black")
        out.append(canvas.quantize(colors=128))
    path = os.path.join(ROOT, "docs", "media", "demo.gif")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out[0].save(path, save_all=True, append_images=out[1:], duration=50, loop=0, optimize=True)
    print(path, len(out), "frames", os.path.getsize(path) // 1024, "KB",
          {j: (len(c[0]) - 1, c[1], c[2]) for j, c in clips.items()})


if __name__ == "__main__":
    main()
