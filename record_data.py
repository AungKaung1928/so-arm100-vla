"""Record the multitask demonstration set through the bench's expert.

    python record_data.py --root data/multitask --episodes 20

Every training (task, colour) combination, all three cubes on the table,
successful episodes only, instructions from the training templates. The
held-out paraphrases and held-out combinations are never recorded: the
evaluation splits stay unseen by construction, not by promise.
"""
import argparse

from so_arm100_sim import instructions as I

from so_arm100_vla.data import record_multitask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multitask")
    ap.add_argument("--repo-id", default="local/so_arm100_multitask")
    ap.add_argument("--episodes", type=int, default=20, help="successful episodes per combo")
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combos", nargs="*", default=None,
                    help="task:colour pairs; default = every training combo")
    a = ap.parse_args()
    combos = None
    if a.combos:
        combos = [tuple(c.split(":")) for c in a.combos]
        bad = [c for c in combos if c in I.HELDOUT_COMBOS]
        if bad:
            raise SystemExit(f"refusing to record held-out combinations: {bad}")
    s = record_multitask(a.root, a.repo_id, a.episodes, combos=combos, seed=a.seed,
                         height=a.height, width=a.width)
    print(f"{s['episodes']} episodes, {s['frames']} frames, {s['wall_s']:.0f} s -> {a.root}")


if __name__ == "__main__":
    main()
