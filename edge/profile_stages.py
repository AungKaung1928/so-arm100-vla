"""Where one SmolVLA chunk's time goes: wall time inside nn.Linear calls, summed by owner
(vision encoder / VLM text layers / action expert / other), PyTorch FP32, 4 threads.
Everything outside Linear calls (attention softmax/matmul, norms, image resize) is 'rest'.

    python edge/profile_stages.py 5
"""
import sys, time, collections
import numpy as np, torch
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
N = int(sys.argv[1])
p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
x = p._get_default_export_input_sample()
def group(name):
    if "vision_model" in name: return "vision encoder"
    if "connector" in name: return "connector"
    if "lm_expert" in name: return "action expert"
    if "text_model" in name or "vlm.model" in name: return "VLM text layers"
    return "other"
acc, calls, t0s = collections.defaultdict(float), collections.Counter(), {}
for name, m in p.model.named_modules():
    if isinstance(m, torch.nn.Linear):
        g = group(name)
        def pre(mod, inp, _n=name):
            t0s[_n] = time.perf_counter()
        def post(mod, inp, out, _n=name, _g=g):   # must return None, or it replaces the output
            acc[_g] += time.perf_counter() - t0s[_n]
            calls[_g] += 1
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)
with torch.no_grad():
    p.model(x)
    acc.clear(); calls.clear()
    walls = []
    for _ in range(N):
        t0 = time.perf_counter(); p.model(x); walls.append(time.perf_counter() - t0)
wall = sum(walls)
print(f"chunk wall p50 {np.median(walls)*1e3:.0f} ms (n {N}, with hooks)")
for g, t in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"{g:16s} {t/N*1e3:7.0f} ms/chunk  {t/wall*100:5.1f} %  Linear calls/chunk {calls[g]//N}")
print(f"{'rest':16s} {(wall-sum(acc.values()))/N*1e3:7.0f} ms/chunk  {(wall-sum(acc.values()))/wall*100:5.1f} %")
