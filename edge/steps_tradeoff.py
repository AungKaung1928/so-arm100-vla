"""Denoising steps vs action error and latency, smolvla_base, PyTorch FP32, 4 threads.
Same inputs for every step count; error is against the default 10-step chunk.

    python edge/steps_tradeoff.py 10
"""
import sys, time
import numpy as np, torch
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
N = int(sys.argv[1])
def load(k):
    return SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32", num_steps=k).eval()
def run(p, x):
    with torch.no_grad():
        o = p.model(x)
    o = o[0] if isinstance(o, (tuple, list)) else (next(iter(o.values())) if isinstance(o, dict) else o)
    return o.float().numpy()
ref = load(10)
base = ref._get_default_export_input_sample()
g = torch.Generator().manual_seed(0)
xs = [base] + [{n: (v + 0.1 * torch.randn(v.shape, generator=g) if v.dtype == torch.float32 else v)
                for n, v in base.items()} for _ in range(2)]
refs = [run(ref, x) for x in xs]
print("10-step chunk spread across the 3 inputs:", f"{max(np.abs(r - refs[0]).max() for r in refs):.3e}", flush=True)
for k in (10, 5, 2, 1):
    p = ref if k == 10 else load(k)
    err = [np.abs(run(p, x) - r) for x, r in zip(xs, refs)]
    for _ in range(2): run(p, base)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter(); run(p, base); ts.append(time.perf_counter() - t0)
    print(f"num_steps {k:2d}: max err vs 10-step {max(e.max() for e in err):.3e} mean {np.mean([e.mean() for e in err]):.3e} "
          f"p50 {np.percentile(ts,50)*1e3:.0f} ms p99 {np.percentile(ts,99)*1e3:.0f} ms (n {N})", flush=True)
    if k != 10: del p
