import pytest
import torch
from transformers import AutoTokenizer

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM, NanoModel
from nano_xyz.model import ModelArchitecture


def test_config_round_trip(tmp_path):
    config = NanoConfig(
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        rope_type="default",
    )
    save_dir = tmp_path / "nano"
    config.save_pretrained(save_dir)
    loaded = NanoConfig.from_pretrained(save_dir)
    assert loaded.block_size == config.block_size
    assert loaded.n_head == config.n_head
    assert loaded.rope_type == "default"


def test_forward_and_generate(tmp_path):
    config = NanoConfig(
        block_size=32,
        vocab_size=256,
        n_layer=1,
        n_head=2,
        n_embd=64,
        rope_type="default",
    )
    model = NanoForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (2, 10, config.vocab_size)

    save_dir = tmp_path / "nano-model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    loaded = NanoForCausalLM.from_pretrained(save_dir)
    gen = loaded.generate(torch.randint(0, config.vocab_size, (1, 8)))
    assert gen.shape[1] >= 8


def test_output_attentions_shape():
    config = NanoConfig(
        block_size=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_lcr=False,
        use_gtr=False,
    )
    model = NanoForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    outputs = model(input_ids=input_ids, output_attentions=True)
    assert outputs.attentions is not None
    assert len(outputs.attentions) == config.n_layer
    assert outputs.attentions[0].shape[:2] == (1, config.n_head)


def test_base_model_outputs_hidden_states_and_cache():
    config = NanoConfig(
        block_size=32,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_embd=64,
        use_lcr=False,
        use_gtr=False,
    )
    model = NanoModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    outputs = model(input_ids=input_ids, use_cache=True)
    assert outputs.last_hidden_state.shape == (1, 5, config.n_embd)
    assert isinstance(outputs.past_key_values, list)


def test_hybrid_blocks_force_cache_disable():
    config = NanoConfig(
        block_size=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_lcr=True,
        lcr_block_indices=[1],
    )
    model = NanoForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    outputs = model(input_ids=input_ids, use_cache=True)
    assert outputs.past_key_values is None


def test_optional_blocks_and_long_context():
    config = NanoConfig(
        n_layer=4,
        n_head=2,
        n_embd=64,
        use_lcr=True,
        lcr_block_indices=[1],
        use_gtr=True,
        gtr_block_indices=[3],
        long_context_mode="none",
        rope_type="default",
    )
    settings = config.to_model_settings()
    assert settings.lcr_block_indices == [1]
    assert settings.gtr_block_indices == [3]
    assert not settings.use_yarn
    assert getattr(settings, "sliding_window", None) is None


def test_conflicting_block_indices_raise():
    cfg = NanoConfig(
        n_layer=4,
        lcr_block_indices=[1],
        gtr_block_indices=[1],
    )
    settings = cfg.to_model_settings()
    with pytest.raises(ValueError):
        ModelArchitecture(settings)


def test_config_pad_defaults_to_eos():
    cfg = NanoConfig(pad_token_id=None, eos_token_id=42)
    assert cfg.pad_token_id == 42


def test_prepare_inputs_for_generation_adds_position_ids():
    config = NanoConfig(
        block_size=32,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_embd=64,
        use_lcr=False,
        use_gtr=False,
    )
    model = NanoForCausalLM(config)
    cache_len = 3
    head_dim = config.n_embd // config.n_head
    past = [
        (
            torch.zeros(1, config.n_head, cache_len, head_dim),
            torch.zeros(1, config.n_head, cache_len, head_dim),
        )
    ]
    input_ids = torch.randint(0, config.vocab_size, (1, 2))
    attention_mask = torch.ones(1, 2, dtype=torch.long)
    prepared = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=past,
        attention_mask=attention_mask,
    )
    assert prepared["input_ids"].shape == (1, 1)
    assert "position_ids" in prepared
    assert prepared["position_ids"].shape == (1, 1)
    assert prepared["attention_mask"].shape[1] == cache_len + 2


def test_gradient_checkpointing_toggle_updates_inner_config():
    config = NanoConfig(
        block_size=32,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=32,
        use_lcr=False,
        use_gtr=False,
    )
    model = NanoForCausalLM(config)
    model.gradient_checkpointing_enable()
    assert model.model.inner.config.use_activation_checkpointing is True
    model.gradient_checkpointing_disable()
    assert model.model.inner.config.use_activation_checkpointing is False


def test_allow_hybrid_cache_enables_cache_outputs():
    config = NanoConfig(
        block_size=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        use_lcr=True,
        lcr_block_indices=[0],
        use_gtr=False,
        allow_hybrid_cache=True,
    )
    model = NanoForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    outputs = model(input_ids=input_ids, use_cache=True)
    assert outputs.past_key_values is not None
