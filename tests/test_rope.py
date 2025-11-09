import torch

from nano_xyz.model import RotaryPositionEmbedding, RopeWithYaRN


def test_rope_cache_extends_and_no_nan():
    rope = RotaryPositionEmbedding(dim=32, max_seq_len=8, base=10_000)
    x = torch.randn(1, 2, 4, 32)
    _ = rope(x, seq_start=0)
    # trigger extension
    out = rope(x, seq_start=20)
    assert rope.max_seq_len >= 24
    assert torch.isfinite(out).all()


def test_rope_nan_clamp_with_extreme_base():
    rope = RotaryPositionEmbedding(dim=16, max_seq_len=4, base=1e12)
    noisy = torch.randn(1, 2, 2, 16) * 1e4
    out = rope(noisy, seq_start=1_000_000)
    assert torch.isfinite(out).all()


def test_yarn_scaling_tracks_target_ctx():
    yarn = RopeWithYaRN(
        dim=16,
        base=10000,
        orig_ctx=128,
        target_ctx=512,
        alpha=1.0,
        beta=1.0,
        enabled=True,
    )
    freqs = yarn._frequencies(seq_len=16, start=0)
    assert freqs.shape[0] == 16
    # resizing beyond target updates target_ctx
    yarn.resize_cache(2048)
    assert yarn.target_ctx == 2048
