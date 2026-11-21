"""Laptop check of the notebook before a Kaggle run (one core, under a minute, no model weights).

    python kaggle/preflight.py            # needs lerobot (Python >= 3.12) and so-arm100-sim

1. Runs the notebook's record command with 1 episode for 2 of its pairs.
2. Parses the notebook's lerobot-train arguments with the installed lerobot and validates them
   (reads config.json of lerobot/smolvla_base from the Hub).
3. Loads one batch and checks the keys and the action chunk shape the policy config asks for.
Not covered: the GPU step itself (memory at batch 32, s/step); the notebook measures those.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_notebook as mk  # noqa: E402


def train_command(ds):
    """The argv the notebook's train() builds, captured instead of run."""
    src = next(s for _, s in mk.CELLS if "def train(" in s).split("for BATCH")[0]
    seen = {}

    class Sub:
        STDOUT = None

        @staticmethod
        def run(cmd, **_):
            seen["cmd"] = cmd
            return type("R", (), {"returncode": 0})()

    ns = {"PY": "/venv/bin/python", "DS": ds, "subprocess": Sub}
    exec(src, ns)
    with tempfile.NamedTemporaryFile() as log:
        ns["train"](20, 2, os.path.join(ds, "_no_such_out"), log.name,
                    ["--log_freq=1", "--save_checkpoint=false"])
    return seen["cmd"]


def main():
    import lerobot
    import torch
    import lerobot.scripts.lerobot_train  # noqa: F401  registers the policies as the CLI does
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset

    pinned = next(s for _, s in mk.CELLS if "lerobot[" in s).split("==")[1].split('"')[0]
    assert lerobot.__version__ == pinned, f"installed lerobot {lerobot.__version__}, notebook pins {pinned}"

    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "sim_smolvla")
        pairs = mk.COMBOS.split()
        subprocess.run([sys.executable, "-m", "so_arm100_sim.record", "--root", ds,
                        "--repo-id", "local/sim_smolvla", "--smolvla", "--episodes", "1",
                        "--tasks", pairs[0], pairs[5]], check=True)
        cmd = train_command(ds)
        assert cmd[0] == "/venv/bin/lerobot-train", cmd[0]
        sys.argv = ["lerobot-train", *cmd[1:]]

        @parser.wrap()
        def check(cfg: TrainPipelineConfig):
            cfg.validate()
            batch = next(iter(torch.utils.data.DataLoader(make_dataset(cfg), batch_size=2)))
            missing = {"action", "task", *cfg.policy.input_features} - set(batch)
            assert not missing, missing
            assert batch["action"].shape[1:] == (cfg.policy.chunk_size, 6), batch["action"].shape
            print(f"preflight ok: lerobot {lerobot.__version__}, {len(cmd) - 1} train arguments parse, "
                  f"batch keys present, action chunk {tuple(batch['action'].shape[1:])}")

        check()


if __name__ == "__main__":
    main()
