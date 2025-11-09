import pytest
import torch

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM


def test_attention_mask_shape_validation_raises_on_mismatch():
    cfg = NanoConfig(
        block_size=32,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_embd=32,
        use_lcr=False,
        use_gtr=False,
    )
    model = NanoForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

    # Wrong batch size in attention_mask
    bad_mask = torch.ones(1, 6, dtype=torch.long)
    with pytest.raises(ValueError):
        model(input_ids=input_ids, attention_mask=bad_mask)

    # Wrong dimensionality (3D) in attention_mask
    bad_mask_3d = torch.ones(2, 6, 1, dtype=torch.long)
    with pytest.raises(ValueError):
        model(input_ids=input_ids, attention_mask=bad_mask_3d)

