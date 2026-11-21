"""The floor under the language results: a policy that holds its start pose.

    python eval_floor.py --episodes 20 --seeds 5 --out runs/eval_floor_hold.json

Same protocol as eval_language.py (bench evaluate, `task:colour:distractors`,
nominal physics, the split's (task, colour) cells). The policy reads no
instruction, so `seen` and `paraphrase` (same pairs) score the same by
construction. Whatever this scores, a trained policy gets for free.
Physics only, no rendering, one core.
"""
import argparse

import numpy as np

from so_arm100_sim.evaluate import evaluate

from eval_language import markdown, summarise_split
from so_arm100_vla import splits as S
from so_arm100_vla.policy import write_json


class HoldRunner:
    """Absolute joint targets = the joint angles seen at the first step of the episode."""

    def reset(self):
        self.q = None

    def act(self, obs):
        if self.q is None:
            self.q = np.array(obs["state"][:6], dtype=float)
        return self.q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/eval_floor_hold.json")
    a = ap.parse_args()
    seeds = tuple(range(a.seeds))
    results = {}
    for split in S.SPLITS:
        tmpl, combos = S.split_spec(split)
        cells = {f"{t}:{c}": evaluate(lambda env: HoldRunner(), f"{t}:{c}:distractors",
                                      n_episodes=a.episodes, seeds=seeds,
                                      env_kwargs={"obs_mode": "proprio", "action_mode": "absolute"})
                 for t, c in combos}
        results[split] = summarise_split(split, tmpl, cells, a.episodes, seeds, "nominal")
    write_json(a.out, {"policy": "hold start pose", "splits": results})
    print(markdown(results))


if __name__ == "__main__":
    main()
