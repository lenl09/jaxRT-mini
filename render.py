import jax.numpy as jnp

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
