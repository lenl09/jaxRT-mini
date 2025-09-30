from volume import create_test_volume, get_volume_world_bounds
from camera import generate_rays
from render import sample_volume_along_rays
from visualize import visualize_rays
import matplotlib.pyplot as plt
import jax.numpy as jnp

def main():
    # 1. Create the volume with physical scale
    volume_shape = (128, 128, 128)
    volume = create_test_volume(shape=volume_shape, gaussian=False, blob_radius=36)
    
    # 2. Define world space bounds (volume occupies a 4x4x4 cube centered at origin)
    world_size = 4.0  # Physical size of the volume in world units
    world_bounds = get_volume_world_bounds(volume_shape, world_size)
    
    # 3. Set up camera parameters in world coordinates
    image_shape = (128, 128)  # Output image size (can be changed without affecting scale!)
    
    # Camera positioned outside the volume looking towards center
    camera_pos = jnp.array([0.0, 0.0, -4.0])  # 3 units back from volume center
    view_dir = jnp.array([0.1, 0.2, 1.0])     # Looking towards +z (into volume)
    
    # Camera sensor size determines field of view (independent of image resolution)
    sensor_width = 5.0   # Physical width of camera sensor 
    sensor_height = 5.0  # Physical height of camera sensor
    
    # Ray marching parameters
    near = 0.1   # Start sampling 0.1 units from camera
    far = 8.0    # Stop sampling 8 units from camera (well past volume)
    num_samples = 128
    
    # 4. Generate rays in world coordinates
    origins, directions, t_vals = generate_rays(
        camera_pos, view_dir, image_shape, num_samples,
        sensor_width=sensor_width, sensor_height=sensor_height,
        near=near, far=far
    )
    
    # 5. Render the image using world coordinates
    ray_samples = sample_volume_along_rays(volume, origins, directions, t_vals, world_bounds)
    
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
    plt.plot(ray_samples[image_shape[0]//2, image_shape[1]//2])
    plt.xlabel('Sample index')
    plt.ylabel('Volume value')
    plt.title('Center Ray Samples')
    plt.show() 
    
    # Render final image
    img_sum = jnp.sum(ray_samples, axis=-1)
    
    plt.figure()
    plt.imshow(img_sum, cmap='gray')
    plt.title(f'Volume Rendering (Resolution: {image_shape[0]}x{image_shape[1]})')
    plt.colorbar()
    plt.show()

def test_resolution_independence():
    """
    Test that changing image resolution doesn't affect the scale of the rendered scene.
    """
    print("\n=== Testing Resolution Independence ===")
    
    # Create volume
    volume_shape = (64, 64, 64)
    volume = create_test_volume(shape=volume_shape, gaussian=False, blob_radius=16)
    world_bounds = get_volume_world_bounds(volume_shape, world_size=4.0)
    
    # Fixed camera parameters in world coordinates
    camera_pos = jnp.array([0.0, 0.0, -4.0])
    view_dir = jnp.array([0.1, 0.2, 1.0])
    sensor_width = 5.0
    sensor_height = 5.0
    near, far = 0.1, 8.0
    num_samples = 64
    
    # Test different image resolutions
    resolutions = [(32, 32), (64, 64), (128, 128)]
    
    fig, axes = plt.subplots(1, len(resolutions), figsize=(15, 5))
    
    for i, image_shape in enumerate(resolutions):
        # Generate rays (same physical sensor, different pixel count)
        origins, directions, t_vals = generate_rays(
            camera_pos, view_dir, image_shape, num_samples,
            sensor_width=sensor_width, sensor_height=sensor_height,
            near=near, far=far
        )
        
        # Render
        ray_samples = sample_volume_along_rays(volume, origins, directions, t_vals, world_bounds)
        img_sum = jnp.sum(ray_samples, axis=-1)
        
        # Plot
        im = axes[i].imshow(img_sum, cmap='gray', extent=[-sensor_width/2, sensor_width/2, -sensor_height/2, sensor_height/2])
        axes[i].set_title(f'{image_shape[0]}x{image_shape[1]} pixels')
        axes[i].set_xlabel('World X')
        axes[i].set_ylabel('World Y')
        
        # Print center ray origin to verify it's the same
        center_origin = origins[image_shape[0]//2, image_shape[1]//2]
        print(f"Resolution {image_shape}: Center ray origin = {center_origin}")
    
    plt.tight_layout()
    plt.suptitle('Same Scene at Different Resolutions (Scale Independent)', y=1.02)
    plt.show()

if __name__ == "__main__":
    main()
    test_resolution_independence()
