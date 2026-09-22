#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tier 1 is the test suite (encoders, splits, windowing, model shapes,
# ensembling, LoRA math, the SmolVLA observation mapping against a stub, and
# an end-to-end record -> train -> evaluate smoke on tiny data). Tiers 2-4
# print the commands that produce every measured number; none of them runs
# unattended from here.
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import torch, mujoco, so_arm100_sim, so_arm100_vla' 2>/dev/null; then
  echo "dependencies are not importable with '$PY'. From the repo root:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "    pip install -r requirements.txt && pip install -e ." >&2
  exit 1
fi
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

hr() { printf '\n=== %s ===\n' "$1"; }

hr "1/4  tests"
"$PY" -m pytest tests -q || exit 1

hr "2/4  data and training"
cat <<'MSG'
Record every training (task, colour) combination, all three cubes on the
table, 20 successful expert episodes each (about 9 x 20 x 150 frames at
20 ms/frame of software rendering, ~10 min):

    python record_data.py --root data/multitask --episodes 20

Train, both encoders, 3 seeds each (report all three):

    nice -n 10 python train.py --root data/multitask --encoder minilm --epochs 30 --seed 0
    nice -n 10 python train.py --root data/multitask --encoder hash   --epochs 30 --seed 0
    nice -n 10 python train.py --root data/multitask --encoder minilm --epochs 30 --seed 0 --no-film
MSG

hr "3/4  the three language splits"
cat <<'MSG'
    python eval_language.py --ckpt runs/policy_minilm_s0.pt --episodes 100 --seeds 5
    python eval_language.py --ckpt runs/policy_hash_s0.pt   --episodes 100 --seeds 5
MSG

hr "4/4  references and gates"
cat <<'MSG'
SmolVLA-base zero-shot (downloads ~1 GB once, about 2 s per control step):

    python smolvla_reference.py --dry-run
    python smolvla_reference.py --run --episodes 10

LoRA compute gate (8 threads, one measurement, under a minute):

    OMP_NUM_THREADS=8 nice -n 10 python lora_gate.py
MSG
if [ -f runs/lora_gate.json ]; then
  "$PY" -c 'import json; b=json.load(open("runs/lora_gate.json")); print(f"\nlast gate: {b[\"step_s_best\"]:.2f} s/step on {b[\"threads\"]} threads -> {b[\"verdict\"]}")'
fi
