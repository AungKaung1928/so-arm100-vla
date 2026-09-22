import torch
import torch.nn as nn

from so_arm100_vla.lora import LoRALinear, attach_lora, trainable_parameters, total_parameters


def test_lora_equals_base_at_init():
    torch.manual_seed(0)
    base = nn.Linear(12, 7)
    x = torch.randn(5, 12)
    ref = base(x).detach().clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    assert torch.allclose(lora(x), ref)
    assert not lora.base.weight.requires_grad and lora.A.requires_grad and lora.B.requires_grad


def test_merge_unmerge_exact():
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(12, 7), r=4, alpha=8.0)
    with torch.no_grad():
        lora.B.normal_()
    x = torch.randn(3, 12)
    w0 = lora.base.weight.detach().clone()
    y = lora(x).detach().clone()
    lora.merge()
    assert torch.allclose(lora(x), y, atol=1e-6)
    assert not torch.allclose(lora.base.weight, w0)
    lora.unmerge()
    assert torch.allclose(lora.base.weight, w0, atol=1e-6)
    assert torch.allclose(lora(x), y, atol=1e-6)


def test_attach_wraps_by_pattern_and_counts_trainables():
    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj = nn.Linear(16, 16), nn.Linear(16, 16), nn.Linear(16, 16)

        def forward(self, x):
            return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
    m = nn.Sequential(Blk(), Blk())
    names = attach_lora(m, r"(q_proj|v_proj)$", r=2, alpha=4.0)
    assert names == ["0.q_proj", "0.v_proj", "1.q_proj", "1.v_proj"]
    assert isinstance(m[0].q_proj, LoRALinear) and isinstance(m[0].k_proj, nn.Linear)
    assert trainable_parameters(m) == 4 * (2 * 16 + 16 * 2)
    assert total_parameters(m) == 6 * (16 * 16 + 16) + trainable_parameters(m)
    y = m(torch.randn(2, 16))
    assert y.shape == (2, 16)
