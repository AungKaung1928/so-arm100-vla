"""INT8 weight-only compression of the exported SmolVLA OpenVINO model (nncf.compress_weights,
default mode INT8_ASYM). Then parity vs PyTorch + latency, same protocol as parity_time.py.

    python edge/int8_ov.py <fp32 dir>/smolvla.xml <int8 dir> 30
"""
import os, sys, time
import numpy as np, torch, openvino as ov, nncf
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
src, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
core = ov.Core()
t0 = time.time()
m = nncf.compress_weights(core.read_model(src))
dst = os.path.join(out, "smolvla_int8.xml")
ov.save_model(m, dst, compress_to_fp16=False)
print(f"compressed {time.time()-t0:.0f} s, bin {os.path.getsize(dst[:-4] + '.bin')/1e9:.2f} GB", flush=True)
p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
base = p._get_default_export_input_sample()
cm = core.compile_model(dst, "CPU", {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32"})
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
    print(f"parity input {k}: max|torch-int8| {np.abs(a-b).max():.3e} mean {np.abs(a-b).mean():.3e} "
          f"max|a(k)-a(0)| {np.abs(a-a0).max():.3e}", flush=True)
for _ in range(3): run_o(base)
ts = []
for _ in range(N):
    t1 = time.perf_counter(); run_o(base); ts.append(time.perf_counter() - t1)
print(f"openvino int8-weights 4 threads n {N}: p50 {np.percentile(ts,50)*1e3:.0f} ms p99 {np.percentile(ts,99)*1e3:.0f} ms", flush=True)
