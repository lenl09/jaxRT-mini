from volume import create_test_volume
import matplotlib.pyplot as plt
from render import render_center_ray, render_image
import jax.numpy as jnp

def main():
    volume = create_test_volume(gaussian=True)
    print("Volume shape:", volume.shape)
    # Show the central slice
    center = volume.shape[0] // 2
    plt.imshow(volume[center, :, :], cmap='gray')
    plt.title(f'Central Slice (z={center})')
    plt.show()

    # Render the center ray
    ray = render_center_ray(volume)
    plt.figure()
    plt.plot(ray)
    plt.title('Center Ray Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.show()
    avg_value = jnp.mean(ray)
    print("Average value along center ray:", avg_value)

    img = render_image(volume, ray_sampler_fn=jnp.sum)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Rendered Image (Mean Along Rays)')
    plt.show()

if __name__ == "__main__":
    main()
