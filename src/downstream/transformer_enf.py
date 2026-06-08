import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable


class TransformerBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float

    @nn.compact
    def __call__(self, x):
        # Self-attention with pre-norm
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
        x = x + y

        # Feed-forward with pre-norm
        y = nn.LayerNorm()(x)
        mlp_dim = int(self.hidden_size * self.mlp_ratio)
        y = nn.Dense(mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.hidden_size)(y)
        x = x + y
        return x


class TransformerClassifier(nn.Module):
    """Transformer classifier operating on ENF latent point clouds.

    Takes pose (p), context (c), and window (g) arrays as input.
    Combines p and c via a learned projection, processes with self-attention
    blocks, and maps to class logits via global average pooling.

    Args:
        hidden_size: Token embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion factor.
        num_classes: Number of output classes.
    """
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    num_classes: int

    @nn.compact
    def __call__(self, p, c, g):
        # Sinusoidal positional embedding from pose coordinates
        freq_scale = 1.0
        freqs = jnp.arange(1, self.hidden_size // (2 * p.shape[-1]) + 1) * freq_scale
        pos_enc = jnp.concatenate([
            jnp.sin(p[..., None] * freqs),
            jnp.cos(p[..., None] * freqs),
        ], axis=-1).reshape(p.shape[0], p.shape[1], -1)

        # Project context to hidden size and add positional encoding
        x = nn.Dense(self.hidden_size)(c) + nn.Dense(self.hidden_size)(pos_enc)

        # Transformer blocks
        for _ in range(self.depth):
            x = TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(x)

        # Global average pooling → classification head
        x = jnp.mean(x, axis=1)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.num_classes)(x)
