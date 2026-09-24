"""Evaluate a trained policy on the three language splits.

    python eval_language.py --ckpt runs/policy_minilm_s0.pt --episodes 100 --seeds 5

For every (task, colour) combination in a split, the bench's protocol runs
with all three cubes on the table; at each episode's reset the runner draws
one sentence from that split's pool for the combination and holds it for
the episode. Success is the bench's own criterion. Output: JSON per split
and one markdown table -- success +- std over seeds, per split and per task.
`--jobs 8` runs the cells in parallel processes; the numbers do not change.
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

from so_arm100_sim.evaluate import evaluate, provenance
from so_arm100_sim import instructions as I

from so_arm100_vla import splits as S
from so_arm100_vla.data import IMAGE
from so_arm100_vla.policy import LangRunner, load_checkpoint, write_json


def _one_thread():
    """Pool initializer: one thread per worker process, so --jobs N uses N
    cores. torch follows set_num_threads; Mesa's software renderer (llvmpipe,
    the only GL on this machine) starts one thread per core unless
    LP_NUM_THREADS is set before the first GL context, which is created later,
    inside the cell."""
    os.environ["LP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


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
    return summarise_split(split, tmpl, cells, n_episodes, seeds, physics)


def summarise_split(split, tmpl, cells, n_episodes, seeds, physics):
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


def run_cell(ckpt, split, task, color, n_episodes, seeds, physics):
    """One (split, task, colour) cell in its own process. Each cell seeds its
    own env and sentence draw, so the numbers equal a serial run."""
    model, stats, encoder, _ = load_checkpoint(ckpt)
    r = eval_split(model, stats, encoder, split, n_episodes, seeds, physics,
                   log=lambda *_: None, combos=[(task, color)])
    return r["cells"][f"{task}:{color}"]


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
    ap.add_argument("--jobs", type=int, default=1, help="cells evaluated in parallel processes")
    a = ap.parse_args()
    model, stats, encoder, ck = load_checkpoint(a.ckpt)
    stem = a.out or os.path.join("runs", "eval_" + os.path.splitext(os.path.basename(a.ckpt))[0])
    seeds = tuple(range(a.seeds))
    parallel = {}
    if a.jobs > 1:
        jobs = [(a.ckpt, split, t, c, a.episodes, seeds, a.physics)
                for split in a.splits for t, c in S.split_spec(split)[1]]
        with mp.get_context("spawn").Pool(min(a.jobs, len(jobs)), initializer=_one_thread) as pool:
            for j, r in zip(jobs, pool.starmap(run_cell, jobs)):
                parallel.setdefault(j[1], {})[f"{j[2]}:{j[3]}"] = r
    results = {}
    for split in a.splits:
        if a.jobs > 1:
            r = summarise_split(split, S.split_spec(split)[0], parallel[split], a.episodes, seeds, a.physics)
            for key, c in r["cells"].items():
                print(f"  {split:10s} {key:12s} success {c['success_rate']:.2f} +- {c['success_std']:.2f}")
        else:
            r = eval_split(model, stats, encoder, split, a.episodes, seeds, a.physics)
        r["ckpt"] = a.ckpt
        r["encoder"] = ck["encoder"]
        results[split] = r
        write_json(f"{stem}_{split}.json", r)
    print()
    print(markdown(results))
    print(f"\nwrote {stem}_<split>.json")


if __name__ == "__main__":
    main()
