"""Export LangPolicy to ONNX, write fixed inputs + PyTorch reference chunks, run cpp/act_ort.

One policy step as LangRunner.act does it, minus the ensembler: uint8 image -> [0,1] CHW,
state -> (s - mean) / std, model, chunk * std + mean. The language embedding is an input:
the encoder runs once per episode, not per step, so it stays in Python.

    python cpp/bench.py --ckpt runs/policy_hash_s0.pt --out runs/act_cpp.json   # real frames
    python cpp/bench.py --synthetic                                            # CI: random weights + inputs
"""
import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import onnxruntime as ort
import torch

from so_arm100_sim.env import ACT_DIM, PROPRIO_DIM
from so_arm100_vla.data import Stats
from so_arm100_vla.model import LangPolicy, image_to_tensor
from so_arm100_vla.policy import load_checkpoint
from so_arm100_vla.text import EMBED_DIM

H, W = 96, 128
HERE = os.path.dirname(os.path.abspath(__file__))


def inputs_real(encoder, n, seed):
    f = np.load("data/multitask/flat.npz")
    idx = np.sort(np.random.default_rng(seed).choice(len(f["states"]), n, replace=False))
    tasks = [str(t) for t in f["tasks"][idx]]
    return f["images"][idx], f["states"][idx], np.asarray(encoder.embed_all(tasks), np.float32)


def inputs_synthetic(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (n, H, W, 3), dtype=np.uint8),
            rng.normal(size=(n, PROPRIO_DIM)).astype(np.float32),
            rng.normal(size=(n, EMBED_DIM)).astype(np.float32))


def torch_step(model, stats, img, state, lang):
    s = torch.as_tensor(stats.norm_state(state), dtype=torch.float32).unsqueeze(0)
    chunk = model(image_to_tensor(img), s, torch.as_tensor(lang).unsqueeze(0))[0].numpy()
    return stats.denorm_action(chunk).astype(np.float32)


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            return next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        return platform.machine()


def pct(xs, q):
    return float(np.percentile(np.asarray(xs) * 1e3, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/policy_hash_s0.pt")
    ap.add_argument("--synthetic", action="store_true", help="seeded random weights, stats and inputs")
    ap.add_argument("--n", type=int, default=64, help="parity samples")
    ap.add_argument("--n-time", type=int, default=500, help="timed steps, each side")
    ap.add_argument("--rounds", type=int, default=3, help="timing rounds: PyTorch, Python ORT, C++ each")
    ap.add_argument("--tol", type=float, default=1e-4, help="max |C++ - PyTorch| in action units (rad)")
    ap.add_argument("--work", default="out/act_cpp")
    ap.add_argument("--out", default=None, help="JSON result path")
    a = ap.parse_args()
    torch.set_num_threads(1)
    load_before = os.getloadavg()[0]
    os.makedirs(a.work, exist_ok=True)

    if a.synthetic:
        torch.manual_seed(0)
        model = LangPolicy().eval()
        rng = np.random.default_rng(1)
        stats = Stats(rng.normal(size=PROPRIO_DIM), rng.uniform(0.5, 2, PROPRIO_DIM),
                      rng.normal(size=ACT_DIM), rng.uniform(0.5, 2, ACT_DIM))
        imgs, states, langs = inputs_synthetic(a.n, 0)
    else:
        model, stats, encoder, _ = load_checkpoint(a.ckpt)
        imgs, states, langs = inputs_real(encoder, a.n, 0)

    onnx_path = os.path.join(a.work, "policy.onnx")
    # the encoder fast path (_transformer_encoder_layer_fwd, used under no_grad) has no ONNX
    # symbolic; export the plain-op path, keep the fast path for the PyTorch reference and timing
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(1, 3, H, W), torch.zeros(1, PROPRIO_DIM),
                                  torch.zeros(1, EMBED_DIM)), onnx_path,
                          input_names=["image", "state", "lang"], output_names=["chunk"],
                          opset_version=17, dynamo=False)
        torch.backends.mha.set_fastpath_enabled(True)

        ref = np.stack([torch_step(model, stats, imgs[i], states[i], langs[i]) for i in range(a.n)])

    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    def ort_step(i):
        return stats.denorm_action(sess.run(None, {
            "image": image_to_tensor(imgs[i]).numpy(),
            "state": stats.norm_state(states[i])[None].astype(np.float32),
            "lang": langs[i][None]})[0][0]).astype(np.float32)

    py_ort = np.stack([ort_step(i) for i in range(a.n)])

    files = {"images.u8": imgs, "states.f32": states, "lang.f32": langs, "ref.f32": ref,
             "stats.f32": np.concatenate([stats.state_mean, stats.state_std,
                                          stats.action_mean, stats.action_std]).astype(np.float32)}
    for name, arr in files.items():
        np.ascontiguousarray(arr).tofile(os.path.join(a.work, name))
    # rounds alternate PyTorch, Python ORT and C++ ORT so all three both see the same background load
    rounds = []
    for _ in range(a.rounds):
        t_torch = []
        with torch.no_grad():
            for i in range(a.n_time):
                j = i % a.n
                t0 = time.perf_counter()
                torch_step(model, stats, imgs[j], states[j], langs[j])
                t_torch.append(time.perf_counter() - t0)
        t_ort = []
        for i in range(a.n_time):
            t0 = time.perf_counter()
            ort_step(i % a.n)
            t_ort.append(time.perf_counter() - t0)
        run = subprocess.run([os.path.join(HERE, "act_ort"), onnx_path, a.work, str(a.n_time)],
                             capture_output=True, text=True)
        print(run.stderr, end="")
        if run.returncode != 0 or not run.stdout.strip():
            raise SystemExit(f"act_ort failed with exit code {run.returncode}")
        cpp = json.loads(run.stdout)
        rounds.append({"torch_p50": pct(t_torch, 50), "torch_p99": pct(t_torch, 99),
                       "python_ort_p50": pct(t_ort, 50), "python_ort_p99": pct(t_ort, 99),
                       "cpp_p50": cpp["step_ms_p50"], "cpp_p99": cpp["step_ms_p99"],
                       "cpp_max_abs_diff": cpp["max_abs_diff"]})
    res = {
        "source": "synthetic (seeded random weights)" if a.synthetic else a.ckpt,
        "n_parity": a.n, "n_time": a.n_time, "rounds": a.rounds, "threads": 1,
        "onnxruntime": ort.__version__, "torch": torch.__version__,
        "cpu": cpu_name(), "nproc": os.cpu_count(),
        "loadavg_1m_before": load_before,
        "max_abs_python_ort_vs_torch": float(np.abs(py_ort - ref).max()),
        "chunk_abs_max": float(np.abs(ref).max()),
        "loadavg_1m_after": os.getloadavg()[0],
        "ort_version_cpp": cpp["ort_version"],
        "max_abs_cpp_vs_torch": max(r["cpp_max_abs_diff"] for r in rounds),
        "step_ms_rounds": rounds,
        "tol": a.tol,
    }
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    raise SystemExit(0 if res["max_abs_cpp_vs_torch"] <= a.tol else 1)


if __name__ == "__main__":
    main()
