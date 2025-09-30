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
    
def sample_volume_along_rays(volume, origins, directions, t_vals, world_bounds, interpolate=True):
    """
    Sample volume values along rays.
    
    Args:
        volume: (X, Y, Z) volume data
        origins: (H, W, 3) ray origins in world coordinates
        directions: (H, W, 3) ray directions in world coordinates  
        t_vals: (H, W, num_samples) parameter values for sampling along rays
        world_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) world coordinate bounds
        interpolate: if True, use trilinear interpolation; if False, use nearest neighbor
        
    Returns:
        ray_samples: (H, W, num_samples) sampled volume values along each ray
    """
    def sample_ray(origin, direction, t_vals_ray):
        """Sample a single ray through the volume."""
        # Compute 3D positions along the ray: origin + t * direction
        positions = origin[None, :] + t_vals_ray[:, None] * direction[None, :]  # (num_samples, 3)
        
        # Convert world coordinates to volume indices
        volume_indices = world_to_volume_indices(positions, world_bounds, volume.shape)
        
        # Sample volume at these positions
        if interpolate:
            values = sample_volume_trilinear(volume, volume_indices)
        else:
            values = sample_volume_nearest(volume, volume_indices)
        
        return values
    
    # Vectorize over all rays (H, W)
    H, W = origins.shape[:2]
    
    # Flatten spatial dimensions for easier processing
    origins_flat = origins.reshape(-1, 3)           # (H*W, 3)
    directions_flat = directions.reshape(-1, 3)     # (H*W, 3)
    t_vals_flat = t_vals.reshape(-1, t_vals.shape[-1])  # (H*W, num_samples)
    
    # Apply vmap over all rays
    all_samples = jax.vmap(sample_ray)(origins_flat, directions_flat, t_vals_flat)
    
    # Reshape back to (H, W, num_samples)
    ray_samples = all_samples.reshape(H, W, -1)
    
    return ray_samples

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

def alpha_compositing(ray_samples, dt=1.0, absorption_coeff=1.0, emission_coeff=1.0):
    """
    Alpha compositing - the standard volume rendering equation.
    
    Args:
        ray_samples: (H, W, num_samples) - density values along rays
        dt: step size along ray
        absorption_coeff: how much light is absorbed per unit density
        emission_coeff: how much light is emitted per unit density
    
    Returns:
        (H, W) - final pixel intensities
    """
    # Convert density to alpha (opacity)
    alpha = 1.0 - jnp.exp(-absorption_coeff * ray_samples * dt)
    
    # Convert density to emission
    emission = emission_coeff * ray_samples
    
    # Alpha compositing from front to back
    transmittance = jnp.cumprod(1.0 - alpha + 1e-10, axis=-1)
    transmittance = jnp.concatenate([jnp.ones_like(transmittance[..., :1]), transmittance[..., :-1]], axis=-1)
    
    # Final color = sum of (emission * alpha * transmittance)
    color = jnp.sum(emission * alpha * transmittance, axis=-1)
    
    return color

def maximum_intensity_projection(ray_samples):
    """
    MIP - takes the maximum value along each ray.
    Good for highlighting high-density features.
    """
    return jnp.max(ray_samples, axis=-1)

def average_intensity_projection(ray_samples):
    """
    Average - simple mean along each ray.
    """
    return jnp.mean(ray_samples, axis=-1)

def sum_intensity_projection(ray_samples):
    """
    Sum - integrates all values along ray (what you're currently using).
    """
    return jnp.sum(ray_samples, axis=-1)

def x_ray_projection(ray_samples, dt=1.0, attenuation_coeff=1.0):
    """
    X-ray style attenuation - models absorption without emission.
    I = I0 * exp(-integral(density))
    """
    integral = jnp.sum(ray_samples * dt, axis=-1)
    return jnp.exp(-attenuation_coeff * integral)

def transfer_function_mapping(ray_samples, transfer_func=None):
    """
    Apply a transfer function to map density values to color/opacity.
    
    Args:
        ray_samples: raw density values
        transfer_func: function that maps density -> (color, alpha)
    """
    if transfer_func is None:
        # Default: linear mapping with threshold
        def default_transfer(density):
            # Threshold low values, enhance high values
            alpha = jnp.clip((density - 0.1) * 2.0, 0.0, 1.0)
            color = density * alpha
            return color, alpha
        transfer_func = default_transfer
    
    colors, alphas = transfer_func(ray_samples)
    
    # Alpha compositing
    transmittance = jnp.cumprod(1.0 - alphas + 1e-10, axis=-1)
    transmittance = jnp.concatenate([jnp.ones_like(transmittance[..., :1]), transmittance[..., :-1]], axis=-1)
    
    final_color = jnp.sum(colors * alphas * transmittance, axis=-1)
    
    return final_color

def ray_box_intersection(origins, directions, box_min, box_max):
    """
    Compute ray-box intersection for all rays.
    
    Args:
        origins: (H, W, 3) - ray origins
        directions: (H, W, 3) - ray directions (should be normalized)
        box_min: (3,) - minimum corner of bounding box
        box_max: (3,) - maximum corner of bounding box
    
    Returns:
        t_near: (H, W) - near intersection parameter (or inf if no intersection)
        t_far: (H, W) - far intersection parameter (or -inf if no intersection)
        intersects: (H, W) - boolean mask of rays that intersect the box
    """
    # Avoid division by zero
    directions_safe = jnp.where(jnp.abs(directions) < 1e-8, 
                               jnp.sign(directions) * 1e-8, 
                               directions)
    
    # Compute intersection parameters for each axis
    t1 = (box_min - origins) / directions_safe
    t2 = (box_max - origins) / directions_safe
    
    # Ensure t1 <= t2 for each axis
    t_min = jnp.minimum(t1, t2)
    t_max = jnp.maximum(t1, t2)
    
    # Ray intersects box if all t_min <= all t_max
    t_near = jnp.max(t_min, axis=-1)  # Latest entry
    t_far = jnp.min(t_max, axis=-1)   # Earliest exit
    
    # Ray intersects if t_near <= t_far and t_far > 0
    intersects = (t_near <= t_far) & (t_far > 0)
    
    # Set non-intersecting rays to have invalid t values
    t_near = jnp.where(intersects, jnp.maximum(t_near, 0), jnp.inf)
    t_far = jnp.where(intersects, t_far, -jnp.inf)
    
    return t_near, t_far, intersects

def generate_optimized_t_vals(t_near, t_far, intersects, num_samples, original_near=0.0, original_far=1.0):
    """
    Generate t values that are optimized to only sample within the volume.
    
    Args:
        t_near: (H, W) - near intersection with volume
        t_far: (H, W) - far intersection with volume  
        intersects: (H, W) - mask of rays that intersect volume
        num_samples: number of samples per ray
        original_near: original near clipping plane
        original_far: original far clipping plane
        
    Returns:
        t_vals: (H, W, num_samples) - optimized t values for sampling
    """
    H, W = t_near.shape
    
    # For non-intersecting rays, use original range (will sample empty space)
    # For intersecting rays, use the intersection range
    effective_near = jnp.where(intersects, 
                              jnp.maximum(t_near, original_near), 
                              original_near)
    effective_far = jnp.where(intersects, 
                             jnp.minimum(t_far, original_far), 
                             original_far)
    
    # Generate linearly spaced samples for each ray
    t_vals = jnp.linspace(0, 1, num_samples)[None, None, :]  # (1, 1, num_samples)
    
    # Scale to the effective range for each ray
    range_size = effective_far[..., None] - effective_near[..., None]  # (H, W, 1)
    t_vals = effective_near[..., None] + t_vals * range_size  # (H, W, num_samples)
    
    return t_vals

def world_to_volume_indices(positions, world_bounds, volume_shape):
    """
    Convert world coordinates to volume indices.
    
    Args:
        positions: (..., 3) world coordinates
        world_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        volume_shape: (X, Y, Z) shape of volume
        
    Returns:
        indices: (..., 3) volume indices (continuous, for interpolation)
    """
    # Extract bounds
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = world_bounds
    X, Y, Z = volume_shape
    
    # Normalize to [0, 1] and then scale to volume indices
    x_norm = (positions[..., 0] - x_min) / (x_max - x_min)
    y_norm = (positions[..., 1] - y_min) / (y_max - y_min)
    z_norm = (positions[..., 2] - z_min) / (z_max - z_min)
    
    # Convert to volume indices
    x_idx = x_norm * (X - 1)
    y_idx = y_norm * (Y - 1)
    z_idx = z_norm * (Z - 1)
    
    indices = jnp.stack([x_idx, y_idx, z_idx], axis=-1)
    return indices

def sample_volume_trilinear(volume, indices):
    """
    Sample volume using trilinear interpolation.
    
    Args:
        volume: (X, Y, Z) volume data
        indices: (..., 3) continuous indices for sampling
        
    Returns:
        values: (...) interpolated volume values
    """
    # Get volume dimensions
    X, Y, Z = volume.shape
    
    # Extract continuous indices
    x, y, z = indices[..., 0], indices[..., 1], indices[..., 2]
    
    # Get integer parts and fractional parts
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)
    
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    
    # Clamp to volume bounds
    x0 = jnp.clip(x0, 0, X - 1)
    y0 = jnp.clip(y0, 0, Y - 1)
    z0 = jnp.clip(z0, 0, Z - 1)
    x1 = jnp.clip(x1, 0, X - 1)
    y1 = jnp.clip(y1, 0, Y - 1)
    z1 = jnp.clip(z1, 0, Z - 1)
    
    # Get fractional parts
    fx = x - x0.astype(jnp.float32)
    fy = y - y0.astype(jnp.float32)
    fz = z - z0.astype(jnp.float32)
    
    # Sample at 8 corners of the cube
    v000 = volume[x0, y0, z0]
    v001 = volume[x0, y0, z1]
    v010 = volume[x0, y1, z0]
    v011 = volume[x0, y1, z1]
    v100 = volume[x1, y0, z0]
    v101 = volume[x1, y0, z1]
    v110 = volume[x1, y1, z0]
    v111 = volume[x1, y1, z1]
    
    # Trilinear interpolation
    v00 = v000 * (1 - fz) + v001 * fz
    v01 = v010 * (1 - fz) + v011 * fz
    v10 = v100 * (1 - fz) + v101 * fz
    v11 = v110 * (1 - fz) + v111 * fz
    
    v0 = v00 * (1 - fy) + v01 * fy
    v1 = v10 * (1 - fy) + v11 * fy
    
    value = v0 * (1 - fx) + v1 * fx
    
    return value

def sample_volume_nearest(volume, indices):
    """
    Sample volume using nearest neighbor interpolation.
    
    Args:
        volume: (X, Y, Z) volume data
        indices: (..., 3) continuous indices for sampling
        
    Returns:
        values: (...) sampled volume values (nearest neighbor)
    """
    # Get volume dimensions
    X, Y, Z = volume.shape
    
    # Extract continuous indices and round to nearest integers
    x = jnp.round(indices[..., 0]).astype(jnp.int32)
    y = jnp.round(indices[..., 1]).astype(jnp.int32)
    z = jnp.round(indices[..., 2]).astype(jnp.int32)
    
    # Clamp to volume bounds
    x = jnp.clip(x, 0, X - 1)
    y = jnp.clip(y, 0, Y - 1)
    z = jnp.clip(z, 0, Z - 1)
    
    # Sample at nearest neighbor indices
    values = volume[x, y, z]
    
    return values