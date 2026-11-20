"""Camera count and image size vs chunk time, smolvla_base, PyTorch FP32, 4 threads.
One loaded model, variants interleaved round-robin. 'change' = max |chunk - 3-camera 512 px chunk|
(a different input, not an error: fewer cameras means less information).

    python edge/cameras_tradeoff.py 8
"""
import sys, time
import numpy as np, torch
import torch.nn.functional as F
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
N = int(sys.argv[1])
p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
x = p._get_default_export_input_sample()
def cams(k):
    return {**x, "images": x["images"][:k], "image_masks": x["image_masks"][:k]}
def size(px):
    im = x["images"]; c, b = im.shape[:2]
    small = F.interpolate(im.reshape(c * b, *im.shape[2:]), size=(px, px), mode="bilinear", align_corners=False)
    return {**x, "images": small.reshape(c, b, 3, px, px)}
variants = {"3 cams 512": x, "2 cams 512": cams(2), "1 cam 512": cams(1), "3 cams 256": size(256)}
a = lambda o: (o[0] if isinstance(o, (tuple, list)) else o).float().numpy()
ref, ok = None, {}
with torch.no_grad():
    for name, v in variants.items():
        try:
            o = a(p.model(v)); ref = o if ref is None else ref
            ok[name] = (v, np.abs(o - ref).max())
        except Exception as e:
            print(f"{name}: fails: {type(e).__name__}: {str(e)[:150]}", flush=True)
    ts = {n: [] for n in ok}
    for i in range(N):
        order = list(ok) if i % 2 == 0 else list(reversed(list(ok)))
        for n in order:
            t0 = time.perf_counter(); p.model(ok[n][0]); ts[n].append(time.perf_counter() - t0)
for n in ok:
    print(f"{n:11s}: p50 {np.median(ts[n])*1e3:5.0f} ms (min {min(ts[n])*1e3:.0f}, max {max(ts[n])*1e3:.0f}, n {N})  change {ok[n][1]:.3e}", flush=True)
