"""3 cameras, PyTorch FP32 vs OpenVINO FP32, interleaved call by call in one process, same input.

    python edge/fp32_interleaved.py <out_dir>/smolvla.xml 20
"""
import sys, time
import numpy as np, torch, openvino as ov
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
xml, N = sys.argv[1], int(sys.argv[2])
p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
x = p._get_default_export_input_sample()
cm = ov.Core().compile_model(xml, "CPU", {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32"})
def run_t():
    with torch.no_grad():
        o = p.model(x)
    return (o[0] if isinstance(o, (tuple, list)) else o).float().numpy()
runs = {"pytorch fp32": run_t, "openvino fp32": lambda: cm([v.numpy() for v in x.values()])[cm.outputs[0]]}
print(f"max |ov - pytorch| {np.abs(runs['openvino fp32']() - run_t()).max():.3e}", flush=True)
ts = {k: [] for k in runs}
for i in range(N):
    for k in (list(runs) if i % 2 == 0 else list(reversed(list(runs)))):
        t0 = time.perf_counter(); runs[k](); ts[k].append(time.perf_counter() - t0)
for k in runs:
    print(f"{k:14s} 3 cams: p50 {np.median(ts[k])*1e3:5.0f} ms p99 {np.percentile(ts[k],99)*1e3:5.0f} ms (n {N})", flush=True)
