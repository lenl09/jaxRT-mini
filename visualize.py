import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp

def visualize_rays(origins, directions, t_vals, num_rays=10, world_bounds=None):
    """
    Visualize a subset of rays in 3D space.
    origins: (H, W, 3) array
    directions: (H, W, 3) array
    t_vals: (num_samples,) array
    num_rays: number of rays to plot (spread across image)
    world_bounds: optional, volume bounds to visualize
    """
    H, W = origins.shape[:2]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pick evenly spaced rays
    step = max(1, min(H, W) // int(jnp.sqrt(num_rays)))
    xs = jnp.arange(0, H, step)[:int(jnp.sqrt(num_rays))]
    ys = jnp.arange(0, W, step)[:int(jnp.sqrt(num_rays))]
    
    for x in xs:
        for y in ys:
            origin = origins[x, y]
            direction = directions[x, y]
            positions = origin + t_vals[:, None] * direction
            ax.plot(positions[:,0], positions[:,1], positions[:,2], alpha=0.7, linewidth=1)
            ax.scatter(origin[0], origin[1], origin[2], color='red', s=20)  # Mark origin
    
    # Draw volume bounds if provided
    if world_bounds is not None:
        x_min, x_max = world_bounds[0]
        y_min, y_max = world_bounds[1] 
        z_min, z_max = world_bounds[2]
        
        # Draw volume cube wireframe
        from itertools import product
        corners = list(product([x_min, x_max], [y_min, y_max], [z_min, z_max]))
        
        # Draw edges
        edges = [
            # Bottom face
            (0, 1), (1, 3), (3, 2), (2, 0),
            # Top face  
            (4, 5), (5, 7), (7, 6), (6, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', alpha=0.3, linewidth=2)
    
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y') 
    ax.set_zlabel('World Z')
    ax.set_title('Ray Visualization (World Coordinates)')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
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
