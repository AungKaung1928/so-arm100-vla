"""The three evaluation splits, built from the bench's instruction tables.

    seen        training templates  x  training (task, colour) combinations
    paraphrase  HELD-OUT templates  x  training combinations
    combo       training templates  x  HELD-OUT combinations

What each proves, if success survives it:

    seen        the policy can execute what it was shown (the floor)
    paraphrase  the instruction is read through the sentence encoder, not
                matched as a template string
    combo       colour and task are composed, not memorised as pairs --
                every task and every colour appeared in training, this
                pairing did not

Sentences and combinations come from `so_arm100_sim.instructions`, so the
bench, the imitation project and this one agree on what "unseen" means.
"""
from so_arm100_sim import instructions as I

SPLITS = ("seen", "paraphrase", "combo")


def split_spec(name):
    """-> (template_split, combos) for one evaluation split."""
    if name == "seen":
        return "train", list(I.TRAIN_COMBOS)
    if name == "paraphrase":
        return "heldout", list(I.TRAIN_COMBOS)
    if name == "combo":
        return "train", list(I.HELDOUT_COMBOS)
    raise KeyError(f"unknown split {name!r}; have {SPLITS}")


def sentences_for(name, task, color):
    tmpl, combos = split_spec(name)
    if (task, color) not in combos:
        raise KeyError(f"({task}, {color}) is not in split {name!r}")
    return I.instructions(task, color, tmpl)


def all_sentences(name):
    tmpl, combos = split_spec(name)
    return [s for _, _, s in I.all_instructions(tmpl, combos)]


def training_sentences():
    """Every sentence the dataset may contain."""
    return all_sentences("seen")
