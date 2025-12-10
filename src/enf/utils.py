import jax
import jax.numpy as jnp
import optax
import math
from typing import Tuple, Any, Type, Dict
from src.enf.bi_invariants import TranslationBI, RotoTranslationBI2D, EmptyBI
import numpy as np
import logging

def initialize_grid_positions(
    batch_size: int, 
    num_latents: int = None,
    latents_per_dim: Tuple[int, ...] = None,  # Optional new parameter
    data_dim: int = None
) -> jnp.ndarray:
    """Initialize a grid of positions in N dimensions.
    
    Args:
        batch_size: Number of samples in batch
        num_latents: Total number of latents (used if latents_per_dim is None)
        latents_per_dim: Optional tuple specifying number of latents for each dimension
                        e.g. (16, 16, 8, 4) for 4D with fewer points in Z and T
        data_dim: Number of dimensions (required if using num_latents)
    """
    if latents_per_dim is not None:
        logging.info(f"Initializing grid positions with latents_per_dim: {latents_per_dim}")
        data_dim = len(latents_per_dim)
        # Create linspace for each dimension with specified number of points
        linspaces = []
        for dim_size in latents_per_dim:
            lims = 1 - 1 / dim_size
            linspaces.append(jnp.linspace(-lims, lims, dim_size))
    else:
        logging.info(f"Initializing grid positions with num_latents: {num_latents} and data_dim: {data_dim}")
        if num_latents is None or data_dim is None:
            raise ValueError("Must provide either latents_per_dim or both num_latents and data_dim")
        # Original even grid logic
        grid_size = int(math.ceil(num_latents ** (1/data_dim)))
        lims = 1 - 1 / grid_size
        linspaces = [jnp.linspace(-lims, lims, grid_size) for _ in range(data_dim)]
    
    # Create meshgrid
    grid = jnp.stack(jnp.meshgrid(*linspaces, indexing='ij'), axis=-1)
    
    # Reshape and repeat for batch
    positions = jnp.reshape(grid, (1, -1, data_dim))
    positions = positions.repeat(batch_size, axis=0)
    
    # If using num_latents and we have more points than needed, truncate
    if latents_per_dim is None and positions.shape[1] > num_latents:
        positions = positions[:, :num_latents, :]
    
    return positions


def initialize_uneven_grid_positions(batch_size: int, num_latents: int, xy_dims: int = 2, z_positions: int = 2) -> jnp.ndarray:
    """Initialize a grid of positions with uneven dimensions.
    
    Args:
        batch_size: Batch size
        num_latents: Total number of latent points
        xy_dims: Number of dimensions for the xy plane (typically 2)
        z_positions: Number of positions in the z dimension (typically 2)
    
    Returns:
        Positions array of shape (batch_size, num_latents, xy_dims + 1)
    """
    
    # Calculate grid size for xy dimensions
    points_per_z = num_latents // z_positions
    grid_size_xy = int(math.ceil(points_per_z ** (1/xy_dims)))
    lims_xy = 1 - 1 / grid_size_xy
    
    # Create linspace for z dimension first (2 positions)
    z_linspace = jnp.linspace(-1, 1, z_positions)
    
    # Create linspaces for xy dimensions
    xy_linspaces = [jnp.linspace(-lims_xy, lims_xy, grid_size_xy) for _ in range(xy_dims)]
    
    # Put z first in the meshgrid inputs to ensure it's the first dimension
    all_linspaces = [z_linspace] + xy_linspaces
    grid = jnp.stack(jnp.meshgrid(*all_linspaces, indexing='ij'), axis=-1)
    
    # Reshape and repeat for batch
    positions = jnp.reshape(grid, (1, -1, xy_dims + 1))
    positions = positions.repeat(batch_size, axis=0)
    
    # If we have more points than needed, truncate
    if positions.shape[1] > num_latents:
        positions = positions[:, :num_latents, :]
    
    return positions


def initialize_latents(
    batch_size: int,
    num_latents: int = None,
    latents_per_dim: Tuple[int, ...] = None,  # Optional new parameter
    latent_dim: int = None,
    data_dim: int = None,
    bi_invariant_cls: Type = None,
    key: Any = None,
    window_scale: float = 2.0,
    noise_scale: float = 0.1,
    z_positions: int = 2,
    even_sampling: bool = True,
    latent_noise: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize the latent variables based on the bi-invariant type.
    
    Args:
        batch_size: Number of samples in batch
        num_latents: Total number of latents (used if latents_per_dim is None)
        latents_per_dim: Optional tuple specifying number of latents for each dimension
                        e.g. (16, 16, 8, 4) for 4D with fewer points in Z and T
        latent_dim: Dimension of the latent space
        data_dim: Number of dimensions of the data space
        bi_invariant_cls: Type of bi-invariant transformation
        key: Random key for initialization
        window_scale: Scale factor for the gaussian window
        noise_scale: Scale factor for the random noise
        z_positions: Number of positions in the z dimension
        even_sampling: Whether to use even grid sampling
        latent_noise: Whether to add random noise to the positions
    
    Returns:
        Tuple of pose, context, and window arrays
    """
    key, subkey = jax.random.split(key)
    
    
    
    # Calculate total number of latents if using latents_per_dim
    total_latents = np.prod(latents_per_dim) if latents_per_dim is not None else num_latents

    if bi_invariant_cls == TranslationBI or bi_invariant_cls == EmptyBI:
        if even_sampling:
            pose = initialize_grid_positions(
                batch_size, 
                num_latents=num_latents,
                latents_per_dim=latents_per_dim,
                data_dim=data_dim
            )
        else:
            pose = initialize_uneven_grid_positions(batch_size, total_latents, z_positions=z_positions)
        
    elif bi_invariant_cls == RotoTranslationBI2D:
        if data_dim != 2:
            raise ValueError("RotoTranslationBI2D requires 2D data")
        
        # Initialize positions in 2D
        positions_2d = initialize_grid_positions(batch_size, total_latents, 2)
        
        # Add orientation angle theta
        key, subkey = jax.random.split(key)
        theta = jax.random.uniform(subkey, (batch_size, total_latents, 1)) * 2 * jnp.pi
        
        # Concatenate positions and theta
        pose = jnp.concatenate([positions_2d, theta], axis=-1)
        
    else:
        raise ValueError(f"Unsupported bi-invariant type: {bi_invariant_cls}")

    # Add random noise to positions
    if latent_noise:
        pose = pose + jax.random.normal(subkey, pose.shape) * noise_scale / jnp.sqrt(total_latents)

    # Initialize context vectors and gaussian window
    context = jnp.ones((batch_size, total_latents, latent_dim)) / latent_dim
    window = jnp.ones((batch_size, total_latents, 1)) * window_scale / jnp.sqrt(total_latents)
    
    return pose, context, window


def initialize_latents_normal(
    batch_size: int,
    num_latents: int,
    latent_dim: int,
    data_dim: int,
    bi_invariant_cls: Type,
    key: Any,
    window_scale: float = 2.0,
    noise_scale: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize the latent variables based on the bi-invariant type."""
    key, subkey = jax.random.split(key)

    if bi_invariant_cls == TranslationBI:
        # For translation-only, positions are same dimension as data
        pose = initialize_grid_positions(batch_size, num_latents, data_dim)
        
    elif bi_invariant_cls == RotoTranslationBI2D:
        if data_dim != 2:
            raise ValueError("RotoTranslationBI2D requires 2D data")
        
        # Initialize positions in 2D
        positions_2d = initialize_grid_positions(batch_size, num_latents, 2)
        
        # Add orientation angle theta
        key, subkey = jax.random.split(key)
        theta = jax.random.uniform(subkey, (batch_size, num_latents, 1)) * 2 * jnp.pi
        
        # Concatenate positions and theta
        pose = jnp.concatenate([positions_2d, theta], axis=-1)
        
    else:
        raise ValueError(f"Unsupported bi-invariant type: {bi_invariant_cls}")

    # Add random noise to positions
    pose = pose + jax.random.normal(subkey, pose.shape) * noise_scale / jnp.sqrt(num_latents)

    # Initialize context vectors and gaussian window
    context = jax.random.normal(subkey, (batch_size, num_latents, latent_dim))
    window = jnp.ones((batch_size, num_latents, 1)) * window_scale / jnp.sqrt(num_latents)
    
    return pose, context, window


def create_coordinate_grid(img_shape: Tuple[int, ...], batch_size: int, num_in: int=2) -> jnp.ndarray:
    """Create a coordinate grid for the input space."""
    
    if num_in == 2:
        x = jnp.stack(jnp.meshgrid(
            jnp.linspace(-1, 1, img_shape[0]),
            jnp.linspace(-1, 1, img_shape[1])), axis=-1)
        x = jnp.reshape(x, (1, -1, 2)).repeat(batch_size, axis=0)
    elif num_in == 3:
        x = jnp.stack(jnp.meshgrid(
            jnp.linspace(-1, 1, img_shape[0]),
            jnp.linspace(-1, 1, img_shape[1]),
            jnp.linspace(-1, 1, img_shape[2]),
            indexing='ij'), axis=-1)
        x = jnp.reshape(x, (1, -1, 3)).repeat(batch_size, axis=0)
    elif num_in == 4:
        x = jnp.stack(jnp.meshgrid(
            jnp.linspace(-1, 1, img_shape[0]),
            jnp.linspace(-1, 1, img_shape[1]),
            jnp.linspace(-1, 1, img_shape[2]),
            jnp.linspace(-1, 1, img_shape[3]),
            indexing='ij'), axis=-1)
        x = jnp.reshape(x, (1, -1, 4)).repeat(batch_size, axis=0)
          
    return x