"""Writes smolvla_finetune.ipynb (Kaggle, GPU T4, Internet on). Run: python kaggle/make_notebook.py

Kept as a script so the notebook is diffable and testable (tests/test_kaggle_notebook.py).
"""
import json
import os

SIM_REV = "4f9b6b1"   # so-arm100-sim commit with `record --smolvla`
COMBOS = " ".join(f"{p}:distractors" for p in (
    "reach:red reach:green reach:blue push:green push:blue "
    "lift:red lift:green pick_place:red pick_place:blue").split())
# instructions.TRAIN_COMBOS, all three cubes on the table as in so_arm100_vla/data.py
# and the bench (eval_language.py), so the colour in the instruction decides the target

MD = f"""# SmolVLA-base fine-tune on so-arm100-sim demonstrations (Kaggle T4)

Settings before **Save Version -> Save & Run All (Commit)**: Accelerator = GPU T4 x2 (one GPU is used), Internet = on.
Steps, expected times and the result check: `kaggle/RUNBOOK.md`.

What runs:
1. Python 3.12 venv (lerobot needs >= 3.12) with `lerobot[smolvla,training]==0.6.1` and so-arm100-sim @ `{SIM_REV}`.
2. Records 20 successful scripted-expert episodes for each of the 9 training (task, colour) pairs,
   with all three cubes on the table (distractors), in the smolvla_base layout (`record --smolvla`: cameras top/front/side -> camera1..3, 6-d state).
   Held-out pairs lift:blue, push:red, pick_place:green are never recorded.
3. Timing run: 20 steps, batch 32 -> 16 -> 8 until it fits; prints s/step (update + data loading).
4. Fine-tune from `lerobot/smolvla_base` for as many steps as fit in `TRAIN_HOURS`.
5. Writes `/kaggle/working/smolvla_sim.zip` (last checkpoint weights + processors, logs). Download it; evaluation runs on the laptop CPU.

The venv, the dataset and the checkpoints live in `/kaggle/temp` (not saved): the saved output folder
`/kaggle/working` is capped at 20 GB and only needs the zip and `train.log`.

Measured before this notebook (laptop CPU, 4 threads, batch 2): 75-86 s/step, 100M of 450M params trainable.
No GPU number exists yet: this notebook produces the first one (label results "Kaggle T4")."""

CELLS = [
    ("code", """TRAIN_HOURS = 2.5     # Kaggle session limit is 12 h; keep margin for recording + zip
EPISODES_PER_PAIR = 20
import os, sys, subprocess
os.environ["MUJOCO_GL"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # T4 x2: one process, one GPU
os.makedirs("/kaggle/temp", exist_ok=True)
print(sys.version)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)"""),
    ("code", f"""%%bash
set -e
pip -q install uv
uv venv -q -p 3.12 /kaggle/temp/venv
uv pip -q install -p /kaggle/temp/venv/bin/python "lerobot[smolvla,training]==0.6.1" \\
  "so-arm100-sim[lerobot] @ git+https://github.com/AungKaung1928/so-arm100-sim@{SIM_REV}"
/kaggle/temp/venv/bin/python -c "import lerobot, so_arm100_sim, torch; print(torch.__version__, torch.cuda.is_available())\""""),
    ("code", """# GL check: EGL first; fall back to OSMesa (apt is fine on Kaggle, not on the laptop)
PY = "/kaggle/temp/venv/bin/python"
def gl_ok(backend):
    env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
    r = subprocess.run([PY, "-c", "from so_arm100_sim import render; print(render.available())"],
                       capture_output=True, text=True, env=env)
    return r.stdout.strip().endswith("True")
if not gl_ok("egl"):
    subprocess.run("apt-get -qq install -y libosmesa6 > /dev/null", shell=True, check=True)
    os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
assert gl_ok(os.environ["MUJOCO_GL"]), "no GL backend"
print("GL:", os.environ["MUJOCO_GL"])"""),
    ("code", f"""import time
DS = "/kaggle/temp/data/sim_smolvla"
t0 = time.time()
subprocess.run([PY, "-m", "so_arm100_sim.record", "--root", DS, "--repo-id", "local/sim_smolvla",
                "--smolvla", "--episodes", str(EPISODES_PER_PAIR), "--tasks", *"{COMBOS}".split()],
               check=True)
print(f"recording wall {{time.time() - t0:.0f}} s")
print(open(DS + "/recording_log.json").read())"""),
    ("code", """import re, shutil
def train(steps, batch, out, log, extra=()):
    cmd = [PY.replace("python", "lerobot-train"), "--policy.path=lerobot/smolvla_base",
           "--dataset.repo_id=local/sim_smolvla", f"--dataset.root={DS}",
           "--policy.push_to_hub=false", "--policy.device=cuda", "--wandb.enable=false",
           f"--batch_size={batch}", f"--steps={steps}", f"--output_dir={out}", *extra]
    with open(log, "w") as f:   # on disk while it runs: the log survives a killed session
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, open(log).read()

def s_per_step(log):
    # one log line per step (--log_freq=1): wall time of a step = update + wait for the batch
    v = [float(u) + float(d) for u, d in re.findall(r"updt_s:([0-9.]+) data_s:([0-9.]+)", log)]
    return sum(v[2:]) / len(v[2:]) if len(v) > 2 else float("nan")   # skip warm-up steps

for BATCH in (32, 16, 8):
    shutil.rmtree("/kaggle/temp/timing", ignore_errors=True)
    rc, log = train(20, BATCH, "/kaggle/temp/timing", "/kaggle/working/timing.log",
                    ["--log_freq=1", "--save_checkpoint=false"])
    if rc == 0:
        break
    print(f"batch {BATCH} failed:", log[-400:])
assert rc == 0, "timing run failed at every batch size: read /kaggle/working/timing.log"
SPS = s_per_step(log)
STEPS = int(TRAIN_HOURS * 3600 / SPS)
print(f"Kaggle T4: batch {BATCH}, {SPS:.3f} s/step (mean of steps 3-20) -> {STEPS} steps in {TRAIN_HOURS} h")"""),
    ("code", """OUT = "/kaggle/temp/train"
shutil.rmtree(OUT, ignore_errors=True)
rc, log = train(STEPS, BATCH, OUT, "/kaggle/working/train.log",
                [f"--save_freq={max(STEPS // 5, 100)}", "--log_freq=50"])
print(rc, log[-1500:])
assert rc == 0, "training failed: read /kaggle/working/train.log\""""),
    ("code", """import zipfile
# Weights + processors only (optimizer state stays on Kaggle). `last` is a symlink: resolve it.
SRC = os.path.realpath(OUT + "/checkpoints/last/pretrained_model")
ZIP = "/kaggle/working/smolvla_sim.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:   # float weights do not compress
    for root, _, files in os.walk(SRC):
        for name in files:
            path = os.path.join(root, name)
            z.write(path, os.path.join("train/checkpoints/last/pretrained_model", os.path.relpath(path, SRC)))
    z.write(DS + "/recording_log.json", "train/recording_log.json")
    z.write("/kaggle/working/train.log", "train/train.log")
names = zipfile.ZipFile(ZIP).namelist()
assert "train/checkpoints/last/pretrained_model/model.safetensors" in names, names
print(len(names), "files,", os.path.getsize(ZIP) // 2**20, "MB ->", ZIP)"""),
]


def build():
    cells = [{"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(True)}]
    for kind, src in CELLS:
        cells.append({"cell_type": kind, "metadata": {}, "execution_count": None,
                      "outputs": [], "source": src.splitlines(True)})
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                        "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smolvla_finetune.ipynb")
    with open(path, "w") as f:
        json.dump(build(), f, indent=1)
    print(path)
