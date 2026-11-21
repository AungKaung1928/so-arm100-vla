"""After the Kaggle run: the downloaded zip, unpacked as RUNBOOK says, loads and evaluates.

Real LeRobot pre/post processors are saved into a checkpoint dir, packed by the notebook's
own zip cell, unpacked to the RUNBOOK path, loaded by eval_smolvla_ft.load and run for one
episode. Only the policy network is a stub (no weights, no download of the VLM).
"""
import os
import re
import shlex
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
E = pytest.importorskip("eval_smolvla_ft")
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla import modeling_smolvla  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402

from test_kaggle_notebook import HERE, mk  # noqa: E402

RUNBOOK = open(os.path.join(HERE, "..", "kaggle", "RUNBOOK.md")).read()


def runbook_paths():
    """Unzip target and --ckpt path from the RUNBOOK's laptop commands."""
    unzip = shlex.split(re.search(r"^.*unzip smolvla_sim\.zip.*$", RUNBOOK, re.M).group(0).split("#")[0])
    block = re.search(r"python eval_smolvla_ft\.py.*?(?=```)", RUNBOOK, re.S).group(0)
    args = shlex.split(block.replace("\\\n", " ").split("#")[0])
    return unzip[unzip.index("-d") + 1], args[args.index("--ckpt") + 1]


def config():
    feats = {"observation.state": PolicyFeature(FeatureType.STATE, (6,))}
    h, w = E.IMAGE["height"], E.IMAGE["width"]
    for key in E.SX.CAMERA_MAP:   # the keys record --smolvla writes (preflight step 4 checks them)
        feats[f"observation.images.{key}"] = PolicyFeature(FeatureType.VISUAL, (3, h, w))
    return SmolVLAConfig(input_features=feats, device="cpu",
                         output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))})


class StubPolicy:
    """Checks what the real preprocessor hands it; returns a normalized zero action."""

    def __init__(self, cfg):
        self.config, self._queues, self.batches = cfg, {"action": []}, 0

    def to(self, _):
        return self

    def eval(self):
        return self

    def reset(self):
        pass

    def select_action(self, batch):
        for key in self.config.input_features:
            assert key in batch, key
        assert "observation.language.tokens" in batch          # tokenizer step ran on the task
        self.batches += 1
        return torch.zeros(1, 6)


@pytest.fixture
def downloaded(tmp_path, monkeypatch):
    cfg = config()
    out, ds, work = tmp_path / "train", tmp_path / "ds", tmp_path / "working"
    model = out / "checkpoints" / "000100" / "pretrained_model"
    model.mkdir(parents=True)
    stats = {"observation.state": {"mean": torch.zeros(6), "std": torch.ones(6)},
             "action": {"mean": torch.full((6,), 10.0), "std": torch.full((6,), 2.0)}}
    try:
        pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    except OSError as e:   # tokenizer not cached and no network
        pytest.skip(f"SmolVLM tokenizer unavailable: {e}")
    pre.save_pretrained(model)          # as lerobot's save_checkpoint does
    post.save_pretrained(model)
    (model / "model.safetensors").write_bytes(b"w")
    (out / "checkpoints" / "last").symlink_to("000100")
    ds.mkdir(); (ds / "recording_log.json").write_text("{}")
    work.mkdir(); (work / "train.log").write_text("step:100")
    src = next("".join(c["source"]) for c in mk.build()["cells"] if "zipfile" in "".join(c["source"]))
    exec(src.replace("/kaggle/working", str(work)), {"os": os, "OUT": str(out), "DS": str(ds)})

    dest, ckpt = runbook_paths()
    monkeypatch.chdir(tmp_path)                       # RUNBOOK paths are relative to the repo root
    zipfile.ZipFile(work / "smolvla_sim.zip").extractall(dest)
    loaded = []

    def from_pretrained(path):
        assert os.path.isfile(os.path.join(path, "model.safetensors")), path
        loaded.append(path)
        return StubPolicy(cfg)

    monkeypatch.setattr(modeling_smolvla.SmolVLAPolicy, "from_pretrained", staticmethod(from_pretrained))
    return ckpt, loaded


def test_runbook_ckpt_path_loads_with_the_real_processors(downloaded):
    ckpt, loaded = downloaded
    policy, pre, post = E.load(ckpt)
    assert loaded == [ckpt]
    a = post(policy.select_action(pre(E.observation(_Env(), {"state": np.zeros(15)}, "lift the cube"))))
    np.testing.assert_allclose(np.asarray(a).reshape(-1), np.full(6, 10.0))   # unnormalized with saved stats


def test_one_episode_smoke_writes_json(downloaded, monkeypatch, tmp_path):
    ckpt, _ = downloaded
    out = tmp_path / "smoke.json"
    real = E.evaluate                                 # 3 steps per episode keeps the test light

    def short(*a, env_kwargs, **k):
        return real(*a, env_kwargs={**env_kwargs, "episode_steps": 3}, **k)

    monkeypatch.setattr(E, "evaluate", short)
    monkeypatch.setattr("sys.argv", ["eval_smolvla_ft.py", "--ckpt", ckpt, "--episodes", "1",
                                     "--splits", "combo", "--out", str(out)])
    E.main()
    import json
    r = json.loads(out.read_text())
    assert set(r["cells"]) == {f"combo/{t}:{c}" for t, c in E.I.HELDOUT_COMBOS}
    assert r["n_forward"] > 0 and r["forward_s_p50"] > 0


class _Env:
    def render(self, cam, h, w):
        return np.zeros((h, w, 3), np.uint8)
