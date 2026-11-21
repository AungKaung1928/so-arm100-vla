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

hr "3b  floor and language probe, rerun against the tracked JSON"
# The hold-still floor needs only the simulator (~90 s, one core). The probe needs the
# recorded demonstrations (data/multitask, ~20 s); without them it is skipped.
"$PY" eval_floor.py --out out/verify_floor.json > /dev/null || exit 1
"$PY" - <<'PY' || exit 1
import json, sys
new, old = (json.load(open(p))["splits"] for p in ("out/verify_floor.json", "runs/eval_floor_hold.json"))
bad = [f"{s}/{c}" for s in old for c in old[s]["cells"]
       if new[s]["cells"][c]["success_rate"] != old[s]["cells"][c]["success_rate"]]
print("floor vs runs/eval_floor_hold.json:", "MISMATCH " + ", ".join(bad) if bad else "match")
sys.exit(1 if bad else 0)
PY
if [ -f data/multitask/flat.npz ]; then
  "$PY" probe_language.py --ckpt runs/policy_minilm_s0.pt --out out/verify_probe.json > /dev/null || exit 1
  "$PY" - <<'PY' || exit 1
import json, sys
new, old = (json.load(open(p)) for p in ("out/verify_probe.json", "runs/probe_language_minilm_s0.json"))
ok = all(new[s] == old[s] for s in ("val", "train"))
print("probe vs runs/probe_language_minilm_s0.json:", "match" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
PY
else
  echo "probe skipped: no data/multitask (record it with step 2)"
fi

hr "4/4  references and gates"
cat <<'MSG'
SmolVLA-base zero-shot (downloads ~1 GB once, about 2 s per control step):

    python smolvla_reference.py --dry-run
    python smolvla_reference.py --run --episodes 10

LoRA compute gate (8 threads, one measurement, under a minute):

    OMP_NUM_THREADS=8 nice -n 10 python lora_gate.py
MSG
if [ -f runs/lora_gate.json ]; then
  "$PY" -c 'import json; b = json.load(open("runs/lora_gate.json")); print("\nlast gate: %.2f s/step on %s threads -> %s" % (b["step_s_best"], b["threads"], b["verdict"]))'
fi
