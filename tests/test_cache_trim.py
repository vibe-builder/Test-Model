import torch

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM


def test_kv_cache_trimming_during_iterative_generation():
    # Small model with explicit small cache limit
    config = NanoConfig(
        block_size=64,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=32,
        use_lcr=False,
        use_gtr=False,
        max_cache_len=10,
    )
    model = NanoForCausalLM(config)
    model.eval()

    # Initial prompt shorter than the cache limit
    prompt_len = 6
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len))

    # First forward to get initial caches
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    past = outputs.past_key_values

    # Sanity: caches exist and are under limit
    assert isinstance(past, list)
    for layer_past in past:
        k, v = layer_past
        assert k.shape[2] <= config.max_cache_len
        assert v.shape[2] <= config.max_cache_len

    # Iteratively decode 25 tokens, ensuring caches are bounded by max_cache_len
    steps = 25
    for _ in range(steps):
        new_token = torch.randint(0, config.vocab_size, (1, 1))
        with torch.no_grad():
            out = model(input_ids=new_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        assert isinstance(past, list)
        for layer_past in past:
            k, v = layer_past
            assert k.shape[2] <= config.max_cache_len
            assert v.shape[2] <= config.max_cache_len

