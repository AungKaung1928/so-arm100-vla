"""Export smolvla_base to OpenVINO FP32 through physicalai's own export path.

    python edge/export_ov.py <out_dir>    (venv from edge/requirements-edge.txt; ~1 min, 4 threads)
"""
import os, sys, time
import torch
torch.set_num_threads(4)
from physicalai.policies.smolvla.policy import SmolVLA
out = sys.argv[1]
t0 = time.time()
p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
print(f"loaded {time.time()-t0:.0f} s, params {sum(x.numel() for x in p.parameters())/1e6:.0f} M", flush=True)
s = p.sample_input
print("sample_input:", {k: (tuple(v.shape), v.dtype) if hasattr(v, "shape") else v for k, v in (s or {}).items()}, flush=True)
t0 = time.time()
p.to_openvino(out)
print(f"exported {time.time()-t0:.0f} s ->", sorted(os.listdir(out)), flush=True)
