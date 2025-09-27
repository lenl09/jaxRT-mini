from volume import create_test_volume
import matplotlib.pyplot as plt
from render import render_center_ray

def main():
    volume = create_test_volume()
    print("Volume shape:", volume.shape)
    # Show the central slice
    center = volume.shape[0] // 2
    plt.imshow(volume[center, :, :], cmap='gray')
    plt.title(f'Central Slice (z={center})')
    plt.show()

    # Render the center ray
    avg_value = render_center_ray(volume)
    print("Average value along center ray:", avg_value)

if __name__ == "__main__":
    main()
