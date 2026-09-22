"""Train the language-conditioned policy on the recorded multitask set.

    nice -n 10 python train.py --root data/multitask --encoder minilm --epochs 30

Reports the final epoch. Validation loss is logged for the curve, never used
to pick a checkpoint. The checkpoint carries the normalisation statistics
and the encoder name, so evaluation cannot silently use different ones.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from so_arm100_vla import text as T
from so_arm100_vla.data import ChunkDataset, Stats, episode_split, load_flat
from so_arm100_vla.model import LangPolicy, image_to_tensor
from so_arm100_vla.policy import save_checkpoint


def to_torch(batch):
    return (image_to_tensor(batch["image"]),
            torch.as_tensor(batch["state"], dtype=torch.float32),
            torch.as_tensor(batch["lang"], dtype=torch.float32),
            torch.as_tensor(batch["actions"], dtype=torch.float32),
            torch.as_tensor(batch["mask"], dtype=torch.float32))


@torch.no_grad()
def evaluate_loss(model, ds, batch_size):
    model.eval()
    tot, n = 0.0, 0
    for b in ds.batches(batch_size, shuffle=False):
        img, st, lg, act, mask = to_torch(b)
        tot += float(LangPolicy.loss(model(img, st, lg), act, mask)) * img.shape[0]
        n += img.shape[0]
    model.train()
    return tot / max(n, 1)


def train(root, repo_id, encoder_name, k=10, d=128, n_kp=32, layers=2, heads=4, film=True,
          epochs=30, batch_size=64, lr=3e-4, seed=0, out="runs/policy.pt", log=print,
          threads=None):
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    flat = load_flat(root, repo_id)
    tr_eps, va_eps = episode_split(flat["episode"], seed=seed)
    stats = Stats.fit(flat["states"][np.isin(flat["episode"], tr_eps)],
                      flat["actions"][np.isin(flat["episode"], tr_eps)])
    encoder = T.make_encoder(encoder_name)
    tr = ChunkDataset(flat, tr_eps, k, encoder, stats)
    va = ChunkDataset(flat, va_eps, k, encoder, stats) if len(va_eps) else None
    cfg = dict(k=k, d=d, n_kp=n_kp, layers=layers, heads=heads, film=film, encoder=encoder_name,
               epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
               train_episodes=int(len(tr_eps)), val_episodes=int(len(va_eps)),
               train_frames=len(tr), unique_sentences=len(tr.embed_of))
    model = LangPolicy(k=k, d=d, n_kp=n_kp, layers=layers, heads=heads, film=film)
    log(f"LangPolicy {model.n_params():,} params, {len(tr)} train windows, "
        f"{len(va) if va else 0} val, {len(tr.embed_of)} unique sentences, encoder {encoder_name}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, (len(tr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    trace = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n, te = 0.0, 0, time.perf_counter()
        for b in tr.batches(batch_size, rng=rng):
            img, st, lg, act, mask = to_torch(b)
            loss = LangPolicy.loss(model(img, st, lg), act, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * img.shape[0]
            n += img.shape[0]
        row = {"epoch": ep, "train_l1": tot / max(n, 1),
               "val_l1": evaluate_loss(model, va, batch_size) if va else None,
               "windows_per_s": n / (time.perf_counter() - te), "elapsed_s": time.perf_counter() - t0}
        trace.append(row)
        log(f"  ep {ep:3d}  train L1 {row['train_l1']:.4f}  "
            f"val L1 {row['val_l1'] if row['val_l1'] is None else round(row['val_l1'], 4)}  "
            f"{row['windows_per_s']:.0f} win/s")
        if len(trace) > 6 and row["windows_per_s"] < 0.8 * np.mean([r["windows_per_s"] for r in trace[:5]]):
            log("  WARNING throughput down >20% from the first epochs: check load average "
                "before reading it as thermal")
    save_checkpoint(out, model, stats, encoder_name, cfg, extra={"trace": trace})
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"cfg": cfg, "trace": trace, "params": model.n_params(),
                   "wall_s": time.perf_counter() - t0}, f, indent=2)
    return model, stats, encoder, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/multitask")
    ap.add_argument("--repo-id", default="local/so_arm100_multitask")
    ap.add_argument("--encoder", choices=("minilm", "hash"), default="minilm")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or f"runs/policy_{a.encoder}{'_nofilm' if a.no_film else ''}_s{a.seed}.pt"
    load1 = float(open("/proc/loadavg").read().split()[0])
    if load1 > 4.0:
        raise SystemExit(f"1-min load average {load1:.2f}: something else is running, not starting")
    train(a.root, a.repo_id, a.encoder, k=a.k, d=a.d, layers=a.layers, film=not a.no_film,
          epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed, out=out,
          threads=a.threads)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
