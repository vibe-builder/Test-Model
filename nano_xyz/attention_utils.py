"""Attention helpers shared across model and generator code."""

from __future__ import annotations

from typing import Optional

import torch


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads."""
    if repeats == 1:
        return hidden_states

    if repeats < 1:
        raise ValueError(f"Repeat factor must be >=1, got {repeats}")

    batch, heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, heads, repeats, seq_len, head_dim)
    return expanded.reshape(batch, heads * repeats, seq_len, head_dim)


def build_attention_mask(
    batch_size: int,
    total_kv_len: int,
    *,
    attention_mask: Optional[torch.Tensor],
    cache_len: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Create an additive mask combining padding info with cached tokens.

    Args:
        batch_size: Number of sequences in the batch.
        total_kv_len: Final key/value sequence length after cache prepended states.
        attention_mask: Optional mask for the *new* tokens (shape: B x new_len).
        cache_len: Number of cached tokens already present in the KV tensor.
        device: Optional device for any tensors we allocate here.

    Returns:
        Additive mask of shape (B, 1, 1, total_kv_len), or None if everything is valid.
    """
    if total_kv_len <= 0:
        return None

    device = device or (attention_mask.device if attention_mask is not None else None)
    if device is None:
        raise ValueError("Device must be provided when attention_mask is None.")

    keep = torch.zeros(batch_size, total_kv_len, device=device, dtype=torch.float32)

    cache_len = int(max(0, min(cache_len, total_kv_len)))
    if cache_len > 0:
        keep[:, :cache_len] = 1.0

    remaining = total_kv_len - cache_len
    if remaining > 0:
        if attention_mask is None:
            new_tokens_mask = torch.ones(batch_size, remaining, device=device, dtype=torch.float32)
        else:
            # Validate attention_mask shape: expect (B, T_new)
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"attention_mask must be 2D (B, T), got {tuple(attention_mask.shape)}"
                )
            if attention_mask.size(0) != batch_size:
                raise ValueError(
                    f"attention_mask batch ({attention_mask.size(0)}) != input batch ({batch_size})"
                )
            new_tokens_mask = attention_mask.to(device=device, dtype=torch.float32)
            copy_len = min(new_tokens_mask.size(1), remaining)
            trimmed = new_tokens_mask[:, -copy_len:]
            if copy_len < remaining:
                pad = torch.zeros(batch_size, remaining, device=device, dtype=torch.float32)
                pad[:, -copy_len:] = trimmed
                trimmed = pad
            elif copy_len > remaining:
                trimmed = trimmed[:, -remaining:]
            new_tokens_mask = trimmed
        keep[:, -remaining:] = new_tokens_mask

    if torch.all(keep.eq(1)):
        return None

    additive_mask = (1.0 - keep) * float("-inf")
    return additive_mask.unsqueeze(1).unsqueeze(2)
