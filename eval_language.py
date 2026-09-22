"""Evaluate a trained policy on the three language splits.

    python eval_language.py --ckpt runs/policy_minilm_s0.pt --episodes 100 --seeds 5

For every (task, colour) combination in a split, the bench's protocol runs
with all three cubes on the table; at each episode's reset the runner draws
one sentence from that split's pool for the combination and holds it for
the episode. Success is the bench's own criterion. Output: JSON per split
and one markdown table -- success +- std over seeds, per split and per task.
"""
import argparse
import os

import numpy as np

from so_arm100_sim.evaluate import evaluate, provenance
from so_arm100_sim import instructions as I

from so_arm100_vla import splits as S
from so_arm100_vla.data import IMAGE
from so_arm100_vla.policy import LangRunner, load_checkpoint, write_json


def eval_split(model, stats, encoder, split, n_episodes, seeds, physics="nominal",
               image=IMAGE, log=print, combos=None):
    tmpl, split_combos = S.split_spec(split)
    combos = split_combos if combos is None else [c for c in split_combos if c in combos]
    cells = {}
    for task, color in combos:
        sentences = I.instructions(task, color, tmpl)
        used = []

        def factory(env, _s=sentences, _u=used):
            return LangRunner(model, stats, encoder, _s, seed=len(_u), log_sentences=_u)

        r = evaluate(factory, f"{task}:{color}:distractors", n_episodes=n_episodes, seeds=seeds,
                     physics=physics, env_kwargs={"obs_mode": "proprio", "action_mode": "absolute",
                                                  "image": dict(image)})
        r["sentences_used"] = sorted(set(used))
        cells[f"{task}:{color}"] = r
        log(f"  {split:10s} {task}:{color:6s} success {r['success_rate']:.2f} +- {r['success_std']:.2f}")
    rates = [c["success_rate"] for c in cells.values()]
    stds = [c["success_std"] for c in cells.values()]
    by_task = {}
    for key, c in cells.items():
        by_task.setdefault(key.split(":")[0], []).append(c["success_rate"])
    return {"split": split, "templates": tmpl, "n_episodes": n_episodes, "seeds": list(seeds),
            "physics": physics, "success_rate": float(np.mean(rates)) if rates else None,
            "success_std_mean": float(np.mean(stds)) if stds else None,
            "by_task": {t: float(np.mean(v)) for t, v in by_task.items()},
            "cells": cells, **provenance()}


def markdown(results):
    tasks = sorted({t for r in results.values() for t in r["by_task"]})
    lines = ["| split | success | mean +- over seeds | " + " | ".join(tasks) + " |",
             "|---|---|---|" + "---|" * len(tasks)]
    for name, r in results.items():
        row = [f"{r['by_task'].get(t, float('nan')):.2f}" for t in tasks]
        lines.append(f"| {name} | {r['success_rate']:.3f} | {r['success_std_mean']:.3f} | "
                     + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="*", default=list(S.SPLITS))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--physics", default="nominal")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, stats, encoder, ck = load_checkpoint(a.ckpt)
    stem = a.out or os.path.join("runs", "eval_" + os.path.splitext(os.path.basename(a.ckpt))[0])
    results = {}
    for split in a.splits:
        r = eval_split(model, stats, encoder, split, a.episodes, tuple(range(a.seeds)), a.physics)
        r["ckpt"] = a.ckpt
        r["encoder"] = ck["encoder"]
        results[split] = r
        write_json(f"{stem}_{split}.json", r)
    print()
    print(markdown(results))
    print(f"\nwrote {stem}_<split>.json")


if __name__ == "__main__":
    main()
