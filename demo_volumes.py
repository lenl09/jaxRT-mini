#!/usr/bin/env python3
"""
Demo script to showcase the different volume creation functions.
This script creates and visualizes different volume types without ray tracing.
"""

from volume import create_test_spherical_volume, create_cube_volume
import matplotlib.pyplot as plt
import jax.numpy as jnp

def demo_volume_creation():
    """
    Demo different volume creation functions and show cross-sections.
    """
    print("=== Volume Creation Demo ===")
    
    # Volume parameters
    shape = (64, 64, 64)
    center_slice = shape[2] // 2
    
    # Create different volumes
    print("\n1. Creating spherical volumes...")
    sphere_hard = create_test_spherical_volume(shape=shape, gaussian=False, blob_radius=20)
    sphere_gaussian = create_test_spherical_volume(shape=shape, gaussian=True, blob_radius=20)
    
    print("2. Creating cube volumes...")
    cube_solid = create_cube_volume(shape=shape, cube_size=30, feathered=False)
    cube_feathered = create_cube_volume(shape=shape, cube_size=30, feathered=True, feather_width=8)
    
    print("3. Creating rectangular cube...")
    cube_rect = create_cube_volume(shape=shape, cube_size=(40, 20, 25), feathered=False)
    cube_rect_feathered = create_cube_volume(shape=shape, cube_size=(40, 20, 25), feathered=True, feather_width=6)
    
    # Visualize cross-sections
    volumes = {
        'Sphere (Hard)': sphere_hard,
        'Sphere (Gaussian)': sphere_gaussian,
        'Cube (Solid)': cube_solid,
        'Cube (Feathered)': cube_feathered,
        'Rectangle (Solid)': cube_rect,
        'Rectangle (Feathered)': cube_rect_feathered
    }
    
    # Show Z-slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, volume) in enumerate(volumes.items()):
        slice_data = volume[:, :, center_slice]
        im = axes[i].imshow(slice_data, cmap='viridis', origin='lower')
        axes[i].set_title(f'{name}\n(Z-slice at {center_slice})')
        axes[i].set_xlabel('Y')
        axes[i].set_ylabel('X')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Print volume statistics
        vol_sum = jnp.sum(volume)
        vol_max = jnp.max(volume)
        print(f"{name}: Total volume = {vol_sum:.1f}, Max value = {vol_max:.3f}")
    
    plt.tight_layout()
    plt.suptitle('Volume Cross-Sections (Z-slices)', y=1.02)
    plt.show()
    
    # Show XY, XZ, YZ slices for one volume
    demo_volume = cube_feathered
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY slice (Z=center)
    axes[0].imshow(demo_volume[:, :, center_slice], cmap='viridis', origin='lower')
    axes[0].set_title('XY slice (Z=center)')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    
    # XZ slice (Y=center)  
    axes[1].imshow(demo_volume[:, center_slice, :], cmap='viridis', origin='lower')
    axes[1].set_title('XZ slice (Y=center)')
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('X')
    
    # YZ slice (X=center)
    axes[2].imshow(demo_volume[center_slice, :, :], cmap='viridis', origin='lower')
    axes[2].set_title('YZ slice (X=center)')
    axes[2].set_xlabel('Z')
    axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    plt.suptitle('Feathered Cube - Different Cross-Sections', y=1.02)
    plt.show()

def demo_feathering_effects():
    """
    Demo different feathering widths and their effects.
    """
    print("\n=== Feathering Effects Demo ===")
    
    shape = (64, 64, 64)
    cube_size = 28
    feather_widths = [0, 3, 6, 12]
    
    fig, axes = plt.subplots(1, len(feather_widths), figsize=(16, 4))
    
    for i, feather_w in enumerate(feather_widths):
        if feather_w == 0:
            volume = create_cube_volume(shape=shape, cube_size=cube_size, feathered=False)
            title = f'Solid Cube'
        else:
            volume = create_cube_volume(shape=shape, cube_size=cube_size, feathered=True, feather_width=feather_w)
            title = f'Feather Width = {feather_w}'
        
        slice_data = volume[:, :, shape[2] // 2]
        im = axes[i].imshow(slice_data, cmap='viridis', origin='lower')
        axes[i].set_title(title)
        axes[i].set_xlabel('Y')
        axes[i].set_ylabel('X')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        print(f"Feather width {feather_w}: Volume sum = {jnp.sum(volume):.1f}")
    
    plt.tight_layout()
    plt.suptitle('Effect of Different Feathering Widths', y=1.02)
    plt.show()

if __name__ == "__main__":
    demo_volume_creation()
    demo_feathering_effects()
