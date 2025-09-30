from volume import create_test_spherical_volume, create_cube_volume, get_volume_world_bounds
from camera import generate_rays, look_at, move_camera_along_view, setup_camera_looking_at
from render import (sample_volume_along_rays, alpha_compositing, 
                   maximum_intensity_projection, average_intensity_projection,
                   sum_intensity_projection, x_ray_projection)
from visualize import visualize_rays
import matplotlib.pyplot as plt
import jax.numpy as jnp

def main():
    # 1. Create the volume with physical scale
    volume_shape = (128, 128, 128)
    volume = create_test_spherical_volume(shape=volume_shape, gaussian=False, blob_radius=36)
    
    # 2. Define world space bounds (volume occupies a 4x4x4 cube centered at origin)
    world_size = 4.0  # Physical size of the volume in world units
    world_bounds = get_volume_world_bounds(volume_shape, world_size)
    
    # 3. Set up camera parameters in world coordinates
    image_shape = (512, 512)  # Output image size (can be changed without affecting scale!)
    
    # Camera positioned outside the volume looking towards center
    camera_pos = jnp.array([0.0, 0.0, -4.0])  # 3 units back from volume center
    view_dir = jnp.array([0.1, 0.2, 1.0])     # Looking towards +z (into volume)
    
    # Camera sensor size determines field of view (independent of image resolution)
    sensor_width = 5.0   # Physical width of camera sensor 
    sensor_height = 5.0  # Physical height of camera sensor
    
    # Ray marching parameters
    near = 0.1   # Start sampling 0.1 units from camera
    far = 8.0    # Stop sampling 8 units from camera (well past volume)
    num_samples = 64
    
    # 4. Generate orthographic rays with volume intersection optimization
    ray_result = generate_rays(
        camera_pos, view_dir, image_shape, num_samples,
        sensor_width=sensor_width, sensor_height=sensor_height,
        near=near, far=far, world_bounds=world_bounds
    )
    
    if len(ray_result) == 4:
        origins, directions, t_vals, intersects = ray_result
        print(f"Ray optimization: {jnp.sum(intersects)} / {jnp.prod(jnp.array(image_shape))} rays intersect volume")
    else:
        origins, directions, t_vals = ray_result
        intersects = None
    
    # 5. Render the image using alpha compositing (more realistic)
    # Choose interpolation method: True = trilinear, False = nearest neighbor
    use_interpolation = True  # Change this to False for nearest neighbor sampling
    ray_samples_interp = sample_volume_along_rays(volume, origins, directions, t_vals, world_bounds, interpolate=use_interpolation)
    ray_samples_nearest = sample_volume_along_rays(volume, origins, directions, t_vals, world_bounds, interpolate=False)
    # Compute step size
    dt = (far - near) / num_samples
    
    # Use alpha compositing instead of simple sum
    img_alpha = alpha_compositing(ray_samples_interp, dt=dt, absorption_coeff=0.1, emission_coeff=1.0)
    img_sum_nearest = jnp.sum(ray_samples_nearest, axis=-1)  # Keep old method for comparison
    img_sum_interp = jnp.sum(ray_samples_interp, axis=-1)

    # Debug: print center ray origin in world coordinates
    center_ray_origin = origins[image_shape[0]//2, image_shape[1]//2]
    print(f"Center ray origin (world coords): {center_ray_origin}")
    print(f"Volume world bounds: {world_bounds}")
    print(f"Camera sensor size: {sensor_width} x {sensor_height}")
    print(f"Image resolution: {image_shape}")
    
    # Visualize rays
    visualize_rays(origins, directions, t_vals, num_rays=8, world_bounds=world_bounds)
    
    # Plot center ray samples
    plt.figure()
    plt.plot(ray_samples_interp[image_shape[0]//2, image_shape[1]//2])
    plt.xlabel('Sample index')
    plt.ylabel('Volume value')
    plt.title('Center Ray Samples (Trilinear)')
    plt.show() 
    
    # Plot both methods
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(img_sum_nearest, cmap='gray')
    ax1.set_title('Sum Projection (Nearest)')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(img_sum_interp, cmap='gray')
    ax2.set_title('Sum Projection (Trilinear)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
