import jax.numpy as jnp
import jax
from volume import world_to_volume_coords

def render_center_ray(volume, num_samples=32):
    """
    Casts a ray through the center of the volume along the z-axis.
    Returns the average value along the ray.
    """
    shape = volume.shape
    # Ray origin: center of x, y; start at z=0
    x0, y0 = shape[0] // 2, shape[1] // 2
    # Sample along z-axis
    z_vals = jnp.linspace(0, shape[2] - 1, num_samples)
    indices = jnp.stack([
        jnp.full_like(z_vals, x0).astype(int),
        jnp.full_like(z_vals, y0).astype(int),
        z_vals.astype(int)
    ], axis=1)
    samples = volume[indices[:,0], indices[:,1], indices[:,2]]
    return samples

def sample_volume_along_rays(volume, origins, directions, t_vals, world_bounds):
    """
    For each ray, sample the volume at positions: origin + t * direction
    Uses world coordinates and transforms to volume indices.
    
    Args:
        volume: (D, H, W) volume data
        origins: (H, W, 3) ray origins in world coordinates
        directions: (H, W, 3) ray directions (normalized) 
        t_vals: (num_samples,) sample distances along rays
        world_bounds: volume world space bounds
        
    Returns: 
        samples (H, W, num_samples)
    """
    H, W = origins.shape[:2]
    
    def sample_ray(origin, direction):
        # Compute world positions along the ray
        positions = origin + t_vals[:, None] * direction  # (num_samples, 3)
        
        # Transform to volume coordinates
        volume_coords = world_to_volume_coords(positions, volume.shape, world_bounds)
        
        # Convert to integer indices and clamp to volume bounds
        indices = jnp.clip(
            jnp.round(volume_coords).astype(int), 
            0, 
            jnp.array(volume.shape) - 1
        )
        
        # Sample the volume
        samples = volume[indices[:,0], indices[:,1], indices[:,2]]
        return samples  # Return all samples along the ray
    
    # Vectorize over all rays
    all_samples = jax.vmap(
        lambda o, d: sample_ray(o, d),
        in_axes=(0, 0)
    )(origins.reshape(-1, 3), directions.reshape(-1, 3))
    
    return all_samples.reshape(H, W, len(t_vals))

def render_image(volume, num_samples=32, ray_sampler_fn=None):
    """
    Renders a 2D image by casting rays along the z-axis for each (x, y) pixel.
    ray_sampler_fn: function that takes ray samples and returns a pixel value. Default is averaging.
    """
    shape = volume.shape
    H, W = shape[0], shape[1]
    xs = jnp.arange(H)
    ys = jnp.arange(W)
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')
    z_vals = jnp.linspace(0, shape[2] - 1, num_samples)
    def sample_ray(x, y):
        indices = jnp.stack([
            jnp.full_like(z_vals, x).astype(int),
            jnp.full_like(z_vals, y).astype(int),
            z_vals.astype(int)
        ], axis=1)
        samples = volume[indices[:,0], indices[:,1], indices[:,2]]
        if ray_sampler_fn is None:
            return jnp.mean(samples)
        else:
            return ray_sampler_fn(samples)
    image = jax.vmap(
        lambda x, y: sample_ray(x, y),
        in_axes=(0, 0)
    )(X.flatten(), Y.flatten())
    return image.reshape(H, W)