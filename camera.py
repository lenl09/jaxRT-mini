import jax.numpy as jnp

def generate_rays(camera_pos, view_dir, image_shape, num_samples, 
                 sensor_width=2.0, sensor_height=2.0, near=0.1, far=10.0, fov=60):
    """
    Generate ray origins and directions for each pixel in the image using physical units.
    
    Args:
        camera_pos: (3,) array, camera position in world coordinates
        view_dir: (3,) array, normalized view direction  
        image_shape: (H, W), output image resolution
        num_samples: int, number of samples along each ray
        sensor_width: float, physical width of the camera sensor in world units
        sensor_height: float, physical height of the camera sensor in world units
        near: float, near clipping plane distance
        far: float, far clipping plane distance
        fov: field of view in degrees (currently unused, using orthographic)
    
    Returns: 
        origins (H, W, 3): ray origins in world coordinates
        directions (H, W, 3): ray directions (normalized)
        t_vals (num_samples,): sample distances along rays
    """
    H, W = image_shape
    
    # Create normalized pixel coordinates from -0.5 to 0.5
    x = jnp.linspace(-0.5, 0.5, W)
    y = jnp.linspace(-0.5, 0.5, H)
    X, Y = jnp.meshgrid(x, y, indexing='xy')
    
    # Scale by physical sensor size (independent of resolution)
    X_world = X * sensor_width
    Y_world = Y * sensor_height
    
    # Create orthogonal basis vectors for the camera
    view_dir = view_dir / jnp.linalg.norm(view_dir)  # Ensure normalized

    # Create right and up vectors (assuming world up is [0, 1, 0])
    world_up = jnp.array([0.0, 1.0, 0.0])
    right = jnp.cross(view_dir, world_up)
    right = right / jnp.linalg.norm(right)
    up = jnp.cross(right, view_dir)
    up = up / jnp.linalg.norm(up)
    
    # For orthographic camera, all origins are offset from camera_pos in the image plane
    # Each pixel gets its own origin point on the sensor plane
    origins = (camera_pos[None, None, :] + 
              X_world[:, :, None] * right[None, None, :] + 
              Y_world[:, :, None] * up[None, None, :])
    
    # All rays have the same direction (orthographic projection)
    directions = jnp.tile(view_dir, (H, W, 1))
    
    # Sample distances along rays (in world units)
    t_vals = jnp.linspace(near, far, num_samples)
    
    return origins, directions, t_vals
