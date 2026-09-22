"""Would a LoRA fine-tune of a SmolVLM-sized model even run on this CPU?

    OMP_NUM_THREADS=8 nice -n 10 python lora_gate.py

Builds a stand-in decoder with the shape of SmolVLM-256M's language model
(SmolLM2-135M: hidden 576, 30 layers, 9 heads, MLP 1536), attaches rank-8
LoRA to the query and value projections, and times ONE training step --
forward, backward, optimizer -- at batch 1, sequence 256. No weights are
downloaded; the timing depends on shape, not on values.

Decision rule, fixed before measuring: **GO if a step is under 5 s**, else
NO-GO. At 5 s/step, 2,000 steps is 2.8 hours, which is the most a chunked,
attended CPU run here can carry. A NO-GO drops the fine-tune from the
project; it does not lower the bar.

This measures compute feasibility only. It says nothing about whether a
fine-tune would help.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from so_arm100_vla.lora import attach_lora, trainable_parameters, total_parameters

SHAPE = dict(hidden=576, layers=30, heads=9, mlp=1536, vocab=49152)
GATE_S = 5.0


class Block(nn.Module):
    def __init__(self, h, heads, mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(h, h, bias=False) for _ in range(4))
        self.heads = heads
        self.up, self.gate, self.down = nn.Linear(h, mlp, bias=False), nn.Linear(h, mlp, bias=False), nn.Linear(mlp, h, bias=False)

    def forward(self, x):
        b, t, h = x.shape
        y = self.ln1(x)
        q = self.q_proj(y).view(b, t, self.heads, -1).transpose(1, 2)
        k = self.k_proj(y).view(b, t, self.heads, -1).transpose(1, 2)
        v = self.v_proj(y).view(b, t, self.heads, -1).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, t, h))
        y = self.ln2(x)
        return x + self.down(torch.nn.functional.silu(self.gate(y)) * self.up(y))


class StandIn(nn.Module):
    def __init__(self, hidden, layers, heads, mlp, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, mlp) for _ in range(layers)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def time_step(model, seq_len, batch=1, warm=1, reps=2):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    ids = torch.randint(0, SHAPE["vocab"], (batch, seq_len))
    times = []
    for i in range(warm + reps):
        t0 = time.perf_counter()
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                 ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        if i >= warm:
            times.append(dt)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", type=int, default=SHAPE["layers"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="runs/lora_gate.json")
    a = ap.parse_args()
    threads = torch.get_num_threads()
    shape = dict(SHAPE, layers=a.layers)
    model = StandIn(shape["hidden"], shape["layers"], shape["heads"], shape["mlp"], shape["vocab"])
    wrapped = attach_lora(model, r"(q_proj|v_proj)$", r=a.rank, alpha=2 * a.rank)
    total, trainable = total_parameters(model), trainable_parameters(model)
    print(f"stand-in: {total / 1e6:.1f}M params, LoRA r={a.rank} on {len(wrapped)} linears, "
          f"{trainable / 1e6:.2f}M trainable ({100 * trainable / total:.2f}%), "
          f"{threads} threads, seq {a.seq}")
    times = time_step(model, a.seq, reps=a.reps)
    step = float(min(times))
    verdict = "GO" if step < GATE_S else "NO-GO"
    print(f"one training step: {step:.2f} s (best of {a.reps}; all {[round(t, 2) for t in times]})"
          f"  -> {verdict} (gate {GATE_S:.0f} s)")
    if verdict == "GO":
        print(f"  2,000 steps = {2000 * step / 3600:.1f} h at this rate")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"shape": shape, "rank": a.rank, "seq": a.seq, "threads": threads,
                   "params_total": total, "params_trainable": trainable,
                   "wrapped": len(wrapped), "step_s": times, "step_s_best": step,
                   "gate_s": GATE_S, "verdict": verdict,
                   "date": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
