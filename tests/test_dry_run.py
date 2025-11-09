import torch

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM


def test_dry_run_train_and_generate():
    cfg = NanoConfig(
        block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=32,
        use_lcr=False, use_gtr=False
    )
    model = NanoForCausalLM(cfg)

    # one training step
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(input_ids=input_ids, labels=input_ids)
    loss = out.loss
    assert loss is not None
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    # quick generation
    model.eval()
    with torch.no_grad():
        gen = model.generate(torch.randint(0, cfg.vocab_size, (1, 8)), max_new_tokens=4)
    assert gen.shape[1] >= 8

