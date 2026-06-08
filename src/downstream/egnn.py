import jax
import jax.numpy as jnp
from flax import linen as nn


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant graph neural network layer.

    Updates node features using messages from k-nearest neighbours.
    Coordinate updates are equivariant by construction.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, p, h):
        # p: (B, N, D)  node positions
        # h: (B, N, F)  node features
        B, N, D = p.shape
        _, _, F = h.shape

        # Pairwise relative positions and squared distances
        rel = p[:, :, None, :] - p[:, None, :, :]       # (B, N, N, D)
        sq_dist = jnp.sum(rel ** 2, axis=-1, keepdims=True)  # (B, N, N, 1)

        # Edge features: [h_i, h_j, sq_dist]
        h_i = jnp.broadcast_to(h[:, :, None, :], (B, N, N, F))
        h_j = jnp.broadcast_to(h[:, None, :, :], (B, N, N, F))
        edge_feat = jnp.concatenate([h_i, h_j, sq_dist], axis=-1)

        # Message MLP
        m_ij = nn.Dense(self.hidden_dim)(edge_feat)
        m_ij = nn.silu(m_ij)
        m_ij = nn.Dense(self.hidden_dim)(m_ij)
        m_ij = nn.silu(m_ij)

        # Aggregate messages
        agg = jnp.sum(m_ij, axis=2)  # (B, N, hidden_dim)

        # Node update MLP
        h_new = nn.Dense(self.hidden_dim)(jnp.concatenate([h, agg], axis=-1))
        h_new = nn.silu(h_new)
        h_new = nn.Dense(self.hidden_dim)(h_new)

        # Residual connection (project h if needed)
        if F != self.hidden_dim:
            h = nn.Dense(self.hidden_dim)(h)
        return p, h + h_new


class EGNNClassifier(nn.Module):
    """EGNN-based classifier for ENF latent point clouds.

    Args:
        hidden_dim: Feature dimension for all EGNN layers.
        num_layers: Number of EGNN message-passing layers.
        num_classes: Number of output classes.
        k_neighbors: Kept for API compatibility; full graph is used.
        use_radius: Kept for API compatibility; not applied.
        radius: Kept for API compatibility; not applied.
    """
    hidden_dim: int
    num_layers: int
    num_classes: int
    k_neighbors: int = 8
    use_radius: bool = False
    radius: float = 1.0

    @nn.compact
    def __call__(self, p, c, g):
        # Project context to hidden dimension
        h = nn.Dense(self.hidden_dim)(c)

        # Stack of EGNN layers
        for _ in range(self.num_layers):
            p, h = EGNNLayer(hidden_dim=self.hidden_dim)(p, h)

        # Global mean pooling → classification head
        x = jnp.mean(h, axis=1)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.num_classes)(x)
