import jax.numpy as jnp
import jax

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

def render_image(volume, num_samples=32, ray_sampler_fn=None):
    """
    Renders a 2D image by casting rays along the z-axis for each (x, y) pixel.
    ray_sampler_fn: function that takes ray samples and returns a pixel value.
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