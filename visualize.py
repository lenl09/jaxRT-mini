import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp

def visualize_rays(origins, directions, t_vals, num_rays=8, world_bounds=None):
    """
    Visualize a subset of rays in 3D space.
    
    Args:
        origins: (H, W, 3) ray origins
        directions: (H, W, 3) ray directions
        t_vals: (H, W, num_samples) or (num_samples,) parameter values along rays
        num_rays: number of rays to visualize
        world_bounds: optional world bounds for volume visualization
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    H, W = origins.shape[:2]
    
    # Sample rays to visualize
    ray_indices = []
    step_h = max(1, H // int(jnp.sqrt(num_rays)))
    step_w = max(1, W // int(jnp.sqrt(num_rays)))
    
    for i in range(0, H, step_h):
        for j in range(0, W, step_w):
            ray_indices.append((i, j))
            if len(ray_indices) >= num_rays:
                break
        if len(ray_indices) >= num_rays:
            break
    
    # Plot each ray
    for i, (h, w) in enumerate(ray_indices):
        origin = origins[h, w]
        direction = directions[h, w]
        
        # Get t_vals for this specific ray
        if t_vals.ndim == 3:  # Per-ray optimized t_vals: (H, W, num_samples)
            ray_t_vals = t_vals[h, w]
        else:  # Shared t_vals: (num_samples,)
            ray_t_vals = t_vals
        
        # Compute positions along this ray
        positions = origin[None, :] + ray_t_vals[:, None] * direction[None, :]
        
        # Plot the ray
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               alpha=0.7, linewidth=1, label=f'Ray {i+1}' if i < 5 else "")
        
        # Mark the origin
        ax.scatter(origin[0], origin[1], origin[2], 
                  color='red', s=20, alpha=0.8)
    
    # Draw volume bounds if provided
    if world_bounds is not None:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = world_bounds
        
        # Draw wireframe box
        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        
        # Define the 12 edges of the cube
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        for edge in edges:
            p1, p2 = vertices[edge[0]], vertices[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ray Visualization ({len(ray_indices)} rays)')
    
    # Add legend for first few rays
    if len(ray_indices) <= 5:
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_volume_slice(volume, slice_axis=2, slice_idx=None):
    """
    Visualize a 2D slice of the 3D volume.
    
    Args:
        volume: 3D volume array
        slice_axis: which axis to slice along (0, 1, or 2)
        slice_idx: index of slice (defaults to middle)
    """
    if slice_idx is None:
        slice_idx = volume.shape[slice_axis] // 2
        
    if slice_axis == 0:
        slice_data = volume[slice_idx, :, :]
        title = f'Volume X-slice at index {slice_idx}'
        xlabel, ylabel = 'Y', 'Z'
    elif slice_axis == 1:
        slice_data = volume[:, slice_idx, :]
        title = f'Volume Y-slice at index {slice_idx}'
        xlabel, ylabel = 'X', 'Z'
    else:  # slice_axis == 2
        slice_data = volume[:, :, slice_idx]
        title = f'Volume Z-slice at index {slice_idx}'
        xlabel, ylabel = 'X', 'Y'
    
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data, cmap='gray', origin='lower')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.show()
