"""Low-rank adaptation of nn.Linear, written out.

    y = W x + b + (alpha / r) * B (A x)

A is (r, in) initialised Kaiming-uniform like a linear layer, B is (out, r)
initialised to ZERO, so at attach time the wrapped layer computes exactly
what the base layer computed. The base weight is frozen. `merge()` folds the
update into W for inference; `unmerge()` takes it back out exactly.

Used only by `lora_gate.py`, which measures whether a step of this on a
SmolVLM-sized network is affordable on this CPU. It does not fine-tune
anything.
"""
import math
import re

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r, self.alpha = int(r), float(alpha)
        self.scale = self.alpha / self.r
        self.A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.merged = False

    def delta(self):
        return self.scale * (self.B @ self.A)

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        return y + self.scale * ((x @ self.A.t()) @ self.B.t())

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta()
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta()
            self.merged = False


def attach_lora(model, pattern=r"(q_proj|v_proj)$", r=8, alpha=16.0):
    """Wrap every nn.Linear whose qualified name matches `pattern`. Freezes
    everything else. Returns the list of wrapped names."""
    for p in model.parameters():
        p.requires_grad_(False)
    rx = re.compile(pattern)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and rx.search(full):
                setattr(mod, child_name, LoRALinear(child, r=r, alpha=alpha))
                wrapped.append(full)
    return wrapped


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model):
    return sum(p.numel() for p in model.parameters())
