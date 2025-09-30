import jax.numpy as jnp
from render import ray_box_intersection, generate_optimized_t_vals

def generate_rays(camera_pos, view_dir, image_shape, num_samples, 
                  sensor_width=2.0, sensor_height=2.0, near=0.1, far=10.0, world_bounds=None):
    """
    Generate orthographic rays from camera in world coordinates with optional volume intersection optimization.
    
    Args:
        camera_pos: (3,) camera position in world coordinates
        view_dir: (3,) viewing direction (will be normalized)
        image_shape: (H, W) output image dimensions
        num_samples: number of samples along each ray
        sensor_width: physical width of camera sensor in world units
        sensor_height: physical height of camera sensor in world units
        near: near clipping plane distance
        far: far clipping plane distance
        world_bounds: optional ((3,), (3,)) tuple of (min_bounds, max_bounds) for volume intersection
    
    Returns:
        origins: (H, W, 3) ray origins in world coordinates
        directions: (H, W, 3) ray directions in world coordinates
        t_vals: (H, W, num_samples) parameter values for sampling along rays
        intersects: (H, W) boolean mask of rays that intersect the volume (if world_bounds provided)
    """
    H, W = image_shape
    
    # Normalize view direction
    view_dir = view_dir / jnp.linalg.norm(view_dir)
    
    # Create camera coordinate system
    up = jnp.array([0.0, 1.0, 0.0])  # World up vector
    right = jnp.cross(view_dir, up)
    right = right / jnp.linalg.norm(right)
    up = jnp.cross(right, view_dir)  # Recompute up to be orthogonal
    
    # Create sensor grid in camera space (centered at origin)
    u = jnp.linspace(-sensor_width/2, sensor_width/2, W)
    v = jnp.linspace(-sensor_height/2, sensor_height/2, H)
    uu, vv = jnp.meshgrid(u, v, indexing='xy')
    
    # Orthographic projection: parallel rays starting from sensor plane
    # Ray origins are distributed across the sensor plane
    sensor_offset = uu[..., None] * right[None, None, :] + vv[..., None] * up[None, None, :]
    origins = camera_pos[None, None, :] + sensor_offset
    
    # All rays have the same direction (parallel)
    directions = jnp.broadcast_to(view_dir, (H, W, 3))
    
    # Optimize t_vals based on volume intersection if bounds provided
    if world_bounds is not None:
        # world_bounds format: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        # Convert to box_min, box_max format: (x_min, y_min, z_min), (x_max, y_max, z_max)
        box_min = jnp.array([world_bounds[0][0], world_bounds[1][0], world_bounds[2][0]])
        box_max = jnp.array([world_bounds[0][1], world_bounds[1][1], world_bounds[2][1]])
        t_near, t_far, intersects = ray_box_intersection(origins, directions, box_min, box_max)
        t_vals = generate_optimized_t_vals(t_near, t_far, intersects, num_samples, near, far)
        return origins, directions, t_vals, intersects
    else:
        # Generate standard t values for sampling along rays
        t_vals = jnp.linspace(near, far, num_samples)
        t_vals = jnp.broadcast_to(t_vals, (H, W, num_samples))
        return origins, directions, t_vals

def look_at(camera_pos, target_pos, up_vector=None):
    """
    Calculate view direction to look at a target position.
    
    Args:
        camera_pos: (3,) current camera position
        target_pos: (3,) target position to look at
        up_vector: (3,) world up vector (default: [0, 1, 0])
    
    Returns:
        view_dir: (3,) normalized view direction vector
    """
    if up_vector is None:
        up_vector = jnp.array([0.0, 1.0, 0.0])
    
    # Calculate view direction (from camera to target)
    view_dir = target_pos - camera_pos
    view_dir = view_dir / jnp.linalg.norm(view_dir)
    
    return view_dir

def move_camera_along_view(camera_pos, view_dir, distance):
    """
    Move camera along its view direction by a specified distance.
    Positive distance moves camera forward (towards target), negative moves backward (away from target).
    
    Args:
        camera_pos: (3,) current camera position
        view_dir: (3,) current view direction (should be normalized)
        distance: scalar distance to move (positive = forward/closer, negative = backward/away)
    
    Returns:
        new_camera_pos: (3,) new camera position
    """
    view_dir_normalized = view_dir / jnp.linalg.norm(view_dir)
    # Positive distance moves forward (in direction of view_dir)
    new_camera_pos = camera_pos + distance * view_dir_normalized
    return new_camera_pos

def setup_camera_looking_at(initial_pos, target_pos, distance_offset=0.0):
    """
    Setup camera to look at target, optionally moving along line of sight.
    
    Args:
        initial_pos: (3,) initial camera position
        target_pos: (3,) position to look at
        distance_offset: additional distance to move along line of sight
                        (positive = closer to target, negative = further from target)
    
    Returns:
        camera_pos: (3,) final camera position
        view_dir: (3,) view direction towards target
    """
    # Calculate initial view direction
    view_dir = look_at(initial_pos, target_pos)
    
    # Move camera along line of sight if requested
    if distance_offset != 0.0:
        camera_pos = move_camera_along_view(initial_pos, view_dir, distance_offset)
        # Recalculate view direction to maintain exact look-at
        view_dir = look_at(camera_pos, target_pos)
    else:
        camera_pos = initial_pos
    
    return camera_pos, view_dir
