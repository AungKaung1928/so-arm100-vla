"""INT8 (ONNX Runtime static QDQ) of the exported policy: error against PyTorch and C++ step time.

Runs after cpp/bench.py, on its work dir (policy.onnx, the 64 evaluation frames, PyTorch
reference chunks). Calibration frames come from the same recording but never the 64 evaluated.

    python cpp/int8.py --calib 32 256 --out runs/act_cpp_int8.json
"""
import argparse
import json
import os
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)

from so_arm100_vla.policy import load_checkpoint

H, W, S, E, K, A = 96, 128, 15, 384, 10, 6
HERE = os.path.dirname(os.path.abspath(__file__))


def feed(img, state, lang, sm, ss):
    return {"image": (img.transpose(2, 0, 1)[None] / 255.0).astype(np.float32),
            "state": ((state - sm) / ss)[None].astype(np.float32), "lang": lang[None].astype(np.float32)}


class Reader(CalibrationDataReader):
    def __init__(self, feeds):
        self.it = iter(feeds)

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/policy_hash_s0.pt")
    ap.add_argument("--work", default="out/act_cpp", help="cpp/bench.py work dir")
    ap.add_argument("--calib", type=int, nargs="+", default=[32, 256])
    ap.add_argument("--n-time", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    w = a.work
    st = np.fromfile(f"{w}/stats.f32", np.float32)
    sm, ss, am, asd = st[:S], st[S:2 * S], st[2 * S:2 * S + A], st[2 * S + A:]
    imgs = np.fromfile(f"{w}/images.u8", np.uint8).reshape(-1, H, W, 3)
    n = len(imgs)
    states = np.fromfile(f"{w}/states.f32", np.float32).reshape(n, S)
    langs = np.fromfile(f"{w}/lang.f32", np.float32).reshape(n, E)
    ref = np.fromfile(f"{w}/ref.f32", np.float32).reshape(n, K, A)
    evals = [feed(imgs[i], states[i], langs[i], sm, ss) for i in range(n)]

    # calibration pool: recorded frames not among the evaluated ones (bench.py draws those with seed 0)
    model, _, encoder, _ = load_checkpoint(a.ckpt)
    f = np.load("data/multitask/flat.npz")
    taken = np.sort(np.random.default_rng(0).choice(len(f["states"]), n, replace=False))
    pool = np.setdiff1d(np.arange(len(f["states"])), taken)
    pick = np.random.default_rng(1).permutation(pool)[:max(a.calib)]
    clang = np.asarray(encoder.embed_all([str(t) for t in f["tasks"][pick]]), np.float32)
    calib = [feed(f["images"][j], f["states"][j], clang[i], sm, ss) for i, j in enumerate(pick)]

    head = [nd.name for nd in onnx.load(f"{w}/policy.onnx").graph.node if nd.name.startswith("/head/")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    z_ref = (ref - am) / asd
    rows = []
    for nc in a.calib:
        for name, excl in (("all INT8", []), ("head fp32", head)):
            path = f"{w}/policy_int8_c{nc}_{'head' if excl else 'all'}.onnx"
            quantize_static(f"{w}/policy.onnx", path, Reader(calib[:nc]), quant_format=QuantFormat.QDQ,
                            per_channel=True, activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8, nodes_to_exclude=excl)
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            out = np.stack([sess.run(None, e)[0][0] * asd + am for e in evals])
            err = np.abs(out - ref)
            cpp = subprocess.run([os.path.join(HERE, "act_ort"), path, w, str(a.n_time)],
                                 capture_output=True, text=True, check=True)
            c = json.loads(cpp.stdout)
            rows.append({"calib_frames": nc, "variant": name, "max_abs_rad": float(err.max()),
                         "mean_abs_rad": float(err.mean()), "p99_abs_rad": float(np.percentile(err, 99)),
                         "cpp_max_abs_rad": c["max_abs_diff"],
                         "cpp_step_ms_p50": c["step_ms_p50"], "cpp_step_ms_p99": c["step_ms_p99"]})
            print(rows[-1])
    fp32 = json.loads(subprocess.run([os.path.join(HERE, "act_ort"), f"{w}/policy.onnx", w, str(a.n_time)],
                                     capture_output=True, text=True, check=True).stdout)
    res = {"ckpt": a.ckpt, "n_eval": n, "onnxruntime": ort.__version__, "threads": 1,
           "quant": "static QDQ, per-channel QInt8 weights, QUInt8 activations, MinMax",
           "fp32_cpp_step_ms_p50": fp32["step_ms_p50"],
           "eval_norm_action_range": [float(z_ref.min()), float(z_ref.max())],
           "calib_norm_action_range": {}, "rows": rows}
    with torch.no_grad():
        for nc in a.calib:
            zc = np.concatenate([model(torch.as_tensor(c["image"]), torch.as_tensor(c["state"]),
                                       torch.as_tensor(c["lang"]))[0].numpy() for c in calib[:nc]])
            res["calib_norm_action_range"][str(nc)] = [float(zc.min()), float(zc.max())]
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
