"""KV-cache utilities for dflash.

Provides data structures and helpers for managing key-value caches used
during speculative / draft-model decoding and standard autoregressive
generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class CacheConfig:
    """Configuration for a KV-cache block."""

    max_batch_size: int = 1
    max_seq_len: int = 2048
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class LayerCache:
    """Single-layer KV cache with a fixed pre-allocated buffer.

    The cache stores keys and values for a single transformer layer.
    Entries are appended left-to-right; ``seq_len`` tracks how many
    tokens have been written.
    """

    def __init__(self, cfg: CacheConfig) -> None:
        shape = (cfg.max_batch_size, cfg.num_heads, cfg.max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self.v = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self.seq_len: int = 0

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new key/value slices and return the full accumulated cache.

        Args:
            k: New keys of shape ``(B, H, T, D)``.
            v: New values of shape ``(B, H, T, D)``.

        Returns:
            Tuple of ``(k_cache, v_cache)`` up to the current sequence length.
        """
        t = k.shape[2]
        end = self.seq_len + t
        if end > self.k.shape[2]:
            raise ValueError(
                f"Cache overflow: seq_len={self.seq_len}, incoming={t}, "
                f"max={self.k.shape[2]}"
            )
        self.k[:, :, self.seq_len : end] = k
        self.v[:, :, self.seq_len : end] = v
        self.seq_len = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def rewind(self, n: int) -> None:
        """Roll back the last *n* tokens (used when draft tokens are rejected)."""
        self.seq_len = max(0, self.seq_len - n)

    def reset(self) -> None:
        """Clear the cache without re-allocating the underlying tensors."""
        self.seq_len = 0


class KVCache:
    """Multi-layer KV cache composed of individual :class:`LayerCache` objects.

    Usage::

        cfg = CacheConfig(num_layers=32, ...)
        cache = KVCache(cfg)

        # Inside a model forward pass for layer i:
        k_full, v_full = cache[i].append(k_new, v_new)
    """

    def __init__(self, cfg: CacheConfig) -> None:
        self.cfg = cfg
        self.layers: List[LayerCache] = [LayerCache(cfg) for _ in range(cfg.num_layers)]

    # ------------------------------------------------------------------
    # Convenience access
    # ------------------------------------------------------------------

    def __getitem__(self, layer_idx: int) -> LayerCache:
        return self.layers[layer_idx]

    @property
    def seq_len(self) -> int:
        """Current sequence length (taken from layer 0 as the reference)."""
        return self.layers[0].seq_len if self.layers else 0

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def rewind(self, n: int) -> None:
        """Rewind all layers by *n* tokens."""
        for layer in self.layers:
            layer.rewind(n)

    def reset(self) -> None:
        """Reset all layer caches."""
        for layer in self.layers:
            layer.reset()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KVCache(layers={self.cfg.num_layers}, "
            f"seq_len={self.seq_len}/{self.cfg.max_seq_len}, "
            f"dtype={self.cfg.dtype})"
        )


def make_cache(model_config, *, max_batch_size: int = 1, max_seq_len: int = 2048) -> KVCache:
    """Construct a :class:`KVCache` from a HuggingFace-style model config.

    Supports configs that expose ``num_hidden_layers``, ``num_attention_heads``,
    and ``hidden_size`` attributes (standard for most causal-LM configs).

    Args:
        model_config: A HuggingFace ``PretrainedConfig`` or compatible object.
        max_batch_size: Maximum batch dimension to pre-allocate.
        max_seq_len: Maximum sequence length to pre-allocate.

    Returns:
        A freshly initialised :class:`KVCache`.
    """
    num_layers = getattr(model_config, "num_hidden_layers", 32)
    num_heads = getattr(model_config, "num_key_value_heads",
                        getattr(model_config, "num_attention_heads", 32))
    hidden_size = getattr(model_config, "hidden_size", 4096)
    head_dim = hidden_size // getattr(model_config, "num_attention_heads", 32)

    cfg = CacheConfig(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
    )
    return KVCache(cfg)
