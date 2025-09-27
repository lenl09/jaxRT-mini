import jax.numpy as jnp
import matplotlib.pyplot as plt

def create_test_volume(shape=(32, 32, 32), blob_center=None, blob_radius=6):
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
    # Blob: 1 inside radius, 0 outside
    volume = jnp.where(dist < blob_radius, 1.0, 0.0)
    return volume