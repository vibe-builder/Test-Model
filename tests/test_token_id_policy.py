import pytest
import torch

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM


def test_token_id_policy_train_raises_eval_clamps():
    cfg = NanoConfig(
        block_size=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=16,
        use_lcr=False,
        use_gtr=False,
    )
    model = NanoForCausalLM(cfg)
    # Out-of-range token id
    input_ids = torch.tensor([[0, 1, cfg.vocab_size, 2]])

    # Training: should raise
    model.train()
    with pytest.raises(ValueError):
        model(input_ids=input_ids)

    # Eval: should clamp and run
    model.eval()
    out = model(input_ids=input_ids)
    assert out.logits.shape[0] == 1

