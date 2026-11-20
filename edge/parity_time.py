"""PyTorch vs OpenVINO FP32 parity and latency for smolvla_base, 4 threads.

    python edge/parity_time.py <out_dir>/smolvla.xml 30    (~4 min)
"""
import sys, time
import numpy as np, torch, openvino as ov
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
xml, N = sys.argv[1], int(sys.argv[2])
p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
base = p._get_default_export_input_sample()
print("inputs:", {k: (tuple(v.shape), str(v.dtype)) for k, v in base.items()}, flush=True)
cm = ov.Core().compile_model(xml, "CPU", {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32"})
print("ov inputs:", [i.get_any_name() for i in cm.inputs], "outputs:", [o.get_any_name() for o in cm.outputs], flush=True)
g = torch.Generator().manual_seed(0)
def variant(k):
    if k == 0: return base
    return {n: (v + 0.1 * torch.randn(v.shape, generator=g) if v.dtype == torch.float32 else v) for n, v in base.items()}
def run_t(x):
    with torch.no_grad():
        o = p.model(x)
    o = o[0] if isinstance(o, (tuple, list)) else (next(iter(o.values())) if isinstance(o, dict) else o)
    return o.float().numpy()
def run_o(x):
    return cm([v.numpy() for v in x.values()])[cm.outputs[0]]
a0 = run_t(base)
for k in range(3):
    x = variant(k); a, b = run_t(x), run_o(x)
    print(f"parity input {k}: shape {a.shape} vs {b.shape} max|torch-ov| {np.abs(a - b).max():.3e} torch min/mean/max {a.min():.3f}/{a.mean():.3f}/{a.max():.3f} frac|a|>=0.999 {(np.abs(a)>=0.999).mean():.2f} max|a(k)-a(0)| {np.abs(a-a0).max():.3e} repeat-torch {np.abs(a - run_t(x)).max():.1e}", flush=True)
for name, f in (("torch", run_t), ("openvino", run_o)):
    for _ in range(3): f(base)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter(); f(base); ts.append(time.perf_counter() - t0)
    print(f"{name} 4 threads n {N}: p50 {np.percentile(ts,50)*1e3:.0f} ms p99 {np.percentile(ts,99)*1e3:.0f} ms", flush=True)
