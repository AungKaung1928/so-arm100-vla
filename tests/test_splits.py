import pytest

from so_arm100_sim import instructions as I

from so_arm100_vla import splits as S


def test_three_splits_are_disjoint_in_the_dimension_they_hold_out():
    seen = set(S.all_sentences("seen"))
    para = set(S.all_sentences("paraphrase"))
    combo = set(S.all_sentences("combo"))
    assert seen and para and combo
    assert not seen & para, "paraphrase split shares a sentence with training"
    assert not seen & combo, "combo split shares a sentence with training"
    # combo split uses training TEMPLATES but held-out pairs: every sentence
    # mentions a (task, colour) pair that training never saw together
    _, held = S.split_spec("combo")
    for task, color in held:
        assert (task, color) not in I.TRAIN_COMBOS


def test_training_sentences_cover_every_task_and_colour():
    tmpl, combos = S.split_spec("seen")
    assert {t for t, _ in combos} == set(I.TASKS)
    assert {c for _, c in combos} == {"red", "green", "blue"}


def test_sentences_for_rejects_wrong_combo():
    task, color = I.HELDOUT_COMBOS[0]
    with pytest.raises(KeyError):
        S.sentences_for("seen", task, color)
    assert len(S.sentences_for("combo", task, color)) == 5
    with pytest.raises(KeyError):
        S.split_spec("zero_shot")
