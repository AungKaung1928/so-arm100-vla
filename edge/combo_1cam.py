"""Fastest combination measured: 1 camera at 512 px, OpenVINO FP32 and INT8 weights, vs PyTorch on the
same input. Export uses a 1-camera input sample (OpenVINO shapes are fixed at export).

    python edge/combo_1cam.py <out_dir> 20 [runs/combo_1cam.json]

The optional third argument also writes the printed table as JSON, so README numbers can be test-checked.
"""
import json, os, sys, time
import numpy as np


def summarise(ts, diffs, n):
    """Per-backend p50/p99 in ms (rounded like the printed table) and max |out - pytorch|."""
    return {"n": n, "backends": {k: {"p50_ms": int(round(float(np.median(v)) * 1e3)),
                                     "p99_ms": int(round(float(np.percentile(v, 99)) * 1e3)),
                                     "max_abs_diff_vs_pytorch": float(diffs[k])} for k, v in ts.items()}}


def main():
    import torch, openvino as ov, nncf
    torch.set_num_threads(4)
    from physicalai.policies.smolvla.policy import SmolVLA
    out, N = sys.argv[1], int(sys.argv[2])
    p = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", dtype="float32").eval()
    x = p._get_default_export_input_sample()
    x = {**x, "images": x["images"][:1].contiguous(), "image_masks": x["image_masks"][:1].contiguous()}
    p.to_openvino(out, input_sample=x)
    core = ov.Core()
    fp32 = os.path.join(out, "smolvla.xml"); int8 = os.path.join(out, "smolvla_int8.xml")
    ov.save_model(nncf.compress_weights(core.read_model(fp32)), int8, compress_to_fp16=False)
    cfg = {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32"}
    cms = {"openvino fp32": core.compile_model(fp32, "CPU", cfg), "openvino int8w": core.compile_model(int8, "CPU", cfg)}
    def run_t():
        with torch.no_grad():
            o = p.model(x)
        return (o[0] if isinstance(o, (tuple, list)) else o).float().numpy()
    runs = {"pytorch fp32": run_t, **{k: (lambda c=c: c([v.numpy() for v in x.values()])[c.outputs[0]]) for k, c in cms.items()}}
    ref = run_t()
    ts = {k: [] for k in runs}
    diffs = {}
    for k, f in runs.items():
        diffs[k] = np.abs(f() - ref).max()
        print(f"{k}: max |out - pytorch| {diffs[k]:.3e}", flush=True)
    for i in range(N):
        for k in (list(runs) if i % 2 == 0 else list(reversed(list(runs)))):
            t0 = time.perf_counter(); runs[k](); ts[k].append(time.perf_counter() - t0)
    for k in runs:
        print(f"{k:15s} 1 cam 512 px: p50 {np.median(ts[k])*1e3:5.0f} ms p99 {np.percentile(ts[k],99)*1e3:5.0f} ms (n {N})", flush=True)
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as fh:
            json.dump(summarise(ts, diffs, N), fh, indent=1)


if __name__ == "__main__":
    main()
