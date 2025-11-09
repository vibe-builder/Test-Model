import torch

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM


def test_hybrid_cache_iterative_generation_with_trimming():
    cfg = NanoConfig(
        block_size=64,
        vocab_size=128,
        n_layer=3,
        n_head=2,
        n_embd=32,
        use_lcr=True,
        lcr_block_indices=[1],
        use_gtr=False,
        allow_hybrid_cache=True,
        max_cache_len=10,
    )
    model = NanoForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)

    past = outputs.past_key_values
    assert isinstance(past, list)
    assert len(past) > 0

    # Iterate a few steps and ensure cache stays within bound
    for _ in range(12):
        new_token = torch.randint(0, cfg.vocab_size, (1, 1))
        with torch.no_grad():
            out = model(input_ids=new_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        for k, v in past:
            assert k.shape[2] <= cfg.max_cache_len
            assert v.shape[2] <= cfg.max_cache_len

