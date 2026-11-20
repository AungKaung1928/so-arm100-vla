"""lora_gate.py: the stand-in's parameter counts are derived by hand and match
runs/lora_gate.json (README 176.7M / 0.55M); a tiny step trains LoRA only."""
import json
import os

import torch

import lora_gate as G
from so_arm100_vla.lora import attach_lora, trainable_parameters, total_parameters

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_param_counts_by_hand_and_saved_run():
    s = G.SHAPE
    h, m, L, V, r = s["hidden"], s["mlp"], s["layers"], s["vocab"], 8
    block = 4 * h * h + 3 * h * m + 2 * 2 * h          # q,k,v,o + up,gate,down + 2 LayerNorms
    base = 2 * V * h + L * block + 2 * h                # embed + head (untied) + final LayerNorm
    lora = 2 * L * r * (h + h)                          # A (r x h) + B (h x r) on q_proj, v_proj
    torch.manual_seed(0)
    model = G.StandIn(h, L, s["heads"], m, V)
    assert total_parameters(model) == base
    wrapped = attach_lora(model, r"(q_proj|v_proj)$", r=r, alpha=2 * r)
    saved = json.load(open(os.path.join(ROOT, "runs", "lora_gate.json")))
    assert (len(wrapped), trainable_parameters(model), total_parameters(model)) == \
        (saved["wrapped"], saved["params_trainable"], saved["params_total"]) == (60, lora, base + lora)
    assert (base + lora, lora) == (176_685_696, 552_960)


def test_tiny_step_updates_lora_only():
    torch.manual_seed(0)
    model = G.StandIn(16, 1, 2, 32, 50)
    attach_lora(model, r"(q_proj|v_proj)$", r=2, alpha=4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    vocab, G.SHAPE["vocab"] = G.SHAPE["vocab"], 50
    try:
        times = G.time_step(model, seq_len=8, warm=0, reps=2)
    finally:
        G.SHAPE["vocab"] = vocab
    assert len(times) == 2
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert changed and changed <= trainable
    assert all("q_proj" in n or "v_proj" in n for n in trainable)
