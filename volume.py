import jax.numpy as jnp
import matplotlib.pyplot as plt

def create_test_volume(shape=(32, 32, 32), gaussian=False, blob_center=None, blob_radius=6):
    """
    Creates a 3D volume with a spherical blob in the center.
    shape: tuple of ints, the size of the volume
    blob_center: tuple, location of the blob (defaults to center)
    blob_radius: int, radius of the blob
    """
    if blob_center is None:
        blob_center = tuple(s // 2 for s in shape)
    # Create a grid of coordinates
    x = jnp.arange(shape[0])
    y = jnp.arange(shape[1])
    z = jnp.arange(shape[2])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # Compute distance from center
    dist = jnp.sqrt((X - blob_center[0])**2 + (Y - blob_center[1])**2 + (Z - blob_center[2])**2)
    
    if not gaussian:
        # Blob: 1 inside radius, 0 outside
        volume = jnp.where(dist < blob_radius, 1.0, 0.0)
    else:
        # Gaussian blob with sigma = blob_radius / 4
        sigma = blob_radius / 4
        volume = jnp.exp(-dist**2 / (2 * sigma**2))
    return volume

def world_to_volume_coords(world_pos, volume_shape, world_bounds):
    """
    Transform world coordinates to volume indices.
    
    Args:
        world_pos: (..., 3) array of world coordinates
        volume_shape: (D, H, W) shape of the volume
        world_bounds: ((min_x, max_x), (min_y, max_y), (min_z, max_z)) world space bounds
    
    Returns:
        volume_indices: (..., 3) array of volume indices (can be fractional)
    """
    world_min = jnp.array([world_bounds[0][0], world_bounds[1][0], world_bounds[2][0]])
    world_max = jnp.array([world_bounds[0][1], world_bounds[1][1], world_bounds[2][1]])
    volume_shape_array = jnp.array(volume_shape)
    
    # Normalize world coordinates to [0, 1]
    normalized = (world_pos - world_min) / (world_max - world_min)
    
    # Scale to volume indices [0, shape-1]
    volume_indices = normalized * (volume_shape_array - 1)
    
    return volume_indices

def get_volume_world_bounds(volume_shape, world_size=4.0):
    """
    Get the world space bounds for a volume.
    
    Args:
        volume_shape: (D, H, W) shape of the volume
        world_size: float, size of the volume in world units (cube)
    
    Returns:
        world_bounds: ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    """
    half_size = world_size / 2
    return ((-half_size, half_size), (-half_size, half_size), (-half_size, half_size))