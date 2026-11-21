"""--swap-colour: the policy is told another cube's colour, success still scores the target."""
import pytest

pytest.importorskip("torch")
E = pytest.importorskip("eval_language")
from so_arm100_vla import text as T  # noqa: E402


def test_told_colour_cycles_and_is_identity_without_swap():
    assert [E.told_colour(c, True) for c in ("red", "green", "blue")] == ["green", "blue", "red"]
    assert E.told_colour("blue", False) == "blue"


@pytest.mark.parametrize("swap", [False, True])
def test_eval_split_tells_the_swapped_colour_and_scores_the_target(monkeypatch, swap):
    seen = {}

    def fake_evaluate(factory, task, **kw):
        runner = factory(None)
        seen["task"], seen["sentences"] = task, runner.sentences
        return {"success_rate": 0.0, "success_std": 0.0}

    monkeypatch.setattr(E, "evaluate", fake_evaluate)
    model = type("StubModel", (), {"k": 10})()                 # only the chunk length is read here
    encoder = T.make_encoder("hash")                           # no download, no checkpoint
    r = E.eval_split(model, None, encoder, "seen", 1, (0,), combos=[("lift", "red")],
                     log=lambda *_: None, swap_colour=swap)
    assert seen["task"] == "lift:red:distractors"                         # target unchanged
    told = "green" if swap else "red"
    assert all(told in s and ("red" not in s or not swap) for s in seen["sentences"])
    assert r["cells"]["lift:red"]["told_colour"] == told and r["swap_colour"] is swap
