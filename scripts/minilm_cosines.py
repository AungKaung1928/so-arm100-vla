"""runs/minilm_cosines.json: MiniLM cosine of "lift the green cube" against its held-out
paraphrases, the same colour with another task, and the same task with another colour
(the numbers in docs/RESULTS.md: table row "MiniLM cosine", key point, step text). Needs the MiniLM model or its
embedding cache; one thread, seconds.

    python scripts/minilm_cosines.py
"""
import json
import os

import numpy as np
import torch

from so_arm100_sim import instructions as I
from so_arm100_vla.text import make_encoder

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ANCHOR, OTHER_TASK, OTHER_COLOUR = "lift the green cube", "push the red cube to the target", "lift the blue cube"
EXTRA = "raise the green block into the air"     # free paraphrase quoted in docs/RESULTS.md, not a template


def main():
    torch.set_num_threads(1)
    para = list(I.instructions("lift", "green", "heldout"))
    enc = make_encoder("minilm")
    v = np.asarray(enc.embed_all([ANCHOR, OTHER_TASK, OTHER_COLOUR, EXTRA] + para), dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cos = v[1:] @ v[0]
    res = {"anchor": ANCHOR, "other_task": [OTHER_TASK, float(cos[0])], "other_colour": [OTHER_COLOUR, float(cos[1])], "extra_paraphrase": [EXTRA, float(cos[2])],
           "paraphrases": [[s, float(c)] for s, c in zip(para, cos[3:])]}
    with open(os.path.join(ROOT, "runs", "minilm_cosines.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
