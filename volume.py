import jax.numpy as jnp
import matplotlib.pyplot as plt

def create_test_spherical_volume(shape=(32, 32, 32), gaussian=False, blob_center=None, blob_radius=6):
    """
    Creates a 3D volume with a spherical blob in the center.
    shape: tuple of ints, the size of the volume
    blob_center: tuple, location of the blob (defaults to center)
    blob_radius: int, radius of the blob
    gaussian: bool, if True creates Gaussian blob, else hard sphere
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

def create_cube_volume(shape=(32, 32, 32), cube_center=None, cube_size=12, feathered=False, feather_width=3):
    """
    Creates a 3D volume with a cubic blob in the center.
    
    Args:
        shape: tuple of ints, the size of the volume
        cube_center: tuple, location of the cube center (defaults to center)
        cube_size: int or tuple, size of the cube (single int for uniform cube, or (w, h, d))
        feathered: bool, if True creates feathered edges with square falloff
        feather_width: int, width of feathering region in voxels
    
    Returns:
        volume: 3D array with cube volume
    """
    if cube_center is None:
        cube_center = tuple(s // 2 for s in shape)
    
    # Handle uniform vs non-uniform cube sizes
    if isinstance(cube_size, (int, float)):
        cube_size = (cube_size, cube_size, cube_size)
    
    # Create a grid of coordinates
    x = jnp.arange(shape[0])
    y = jnp.arange(shape[1])
    z = jnp.arange(shape[2])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Compute distance from center along each axis (L-infinity norm)
    dx = jnp.abs(X - cube_center[0])
    dy = jnp.abs(Y - cube_center[1])
    dz = jnp.abs(Z - cube_center[2])
    
    if not feathered:
        # Hard cube: 1 inside, 0 outside
        half_sizes = jnp.array(cube_size) / 2
        volume = jnp.where(
            (dx < half_sizes[0]) & (dy < half_sizes[1]) & (dz < half_sizes[2]),
            1.0, 0.0
        )
    else:
        # Feathered cube with square falloff
        half_sizes = jnp.array(cube_size) / 2
        
        # For each axis, compute the distance-based intensity
        def axis_intensity(dist, half_size, feather_w):
            # Inside the core: full intensity
            core_region = dist < half_size
            # In feather region: linear falloff
            feather_region = (dist >= half_size) & (dist < half_size + feather_w)
            feather_intensity = jnp.maximum(0.0, 1.0 - (dist - half_size) / feather_w)
            
            return jnp.where(core_region, 1.0, 
                           jnp.where(feather_region, feather_intensity, 0.0))
        
        # Compute intensity along each axis
        intensity_x = axis_intensity(dx, half_sizes[0], feather_width)
        intensity_y = axis_intensity(dy, half_sizes[1], feather_width)
        intensity_z = axis_intensity(dz, half_sizes[2], feather_width)
        
        # Combine using multiplication (square falloff pattern)
        volume = intensity_x * intensity_y * intensity_z
    
    return volume

# Keep the old function name for backward compatibility
def create_test_volume(shape=(32, 32, 32), gaussian=False, blob_center=None, blob_radius=6):
    """
    Deprecated: Use create_test_spherical_volume instead.
    Creates a 3D volume with a spherical blob in the center.
    """
    return create_test_spherical_volume(shape, gaussian, blob_center, blob_radius)

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