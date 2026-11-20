"""Settle the step-count timing question: one model, _num_steps switched between calls,
10 and 1 interleaved so drift and noise hit both equally. PyTorch FP32, 4 threads.

    python edge/steps_interleaved.py 10
"""
import sys, time
import numpy as np, torch
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
N = int(sys.argv[1])
p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
x = p._get_default_export_input_sample()
core = p.model._model
assert hasattr(core, "_num_steps"), "attribute moved"
ts = {10: [], 1: []}
out = {}
with torch.no_grad():
    for k in (10, 1):
        core._num_steps = k; out[k] = p.model(x)
    for i in range(N):
        for k in ((10, 1) if i % 2 == 0 else (1, 10)):
            core._num_steps = k
            t0 = time.perf_counter(); p.model(x); ts[k].append(time.perf_counter() - t0)
a = lambda o: (o[0] if isinstance(o, (tuple, list)) else o).float().numpy()
print(f"max |10-step - 1-step| {np.abs(a(out[10]) - a(out[1])).max():.3e} (nonzero = the switch took effect)")
for k in (10, 1):
    print(f"num_steps {k:2d}: p50 {np.median(ts[k])*1e3:.0f} ms min {min(ts[k])*1e3:.0f} max {max(ts[k])*1e3:.0f} (n {N})")
