#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Load Male Lena Image
def load_image(image_path):
    """Load the Male Lena image, convert to grayscale, and normalize."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image, dtype=np.float32) / 255.0

# Generate X-ray Lines from the Very Top
def generate_xray_lines(image_shape, num_lines):
    """Generate X-ray lines originating from the very top edge of the image."""
    height, width = image_shape
    lines = []

    # Divide the image into two vertical regions: left and right halves
    left_start, left_end = 0, width // 2
    right_start, right_end = width // 2, width

    for _ in range(num_lines // 2):
        # Generate random start points at the very top of the image
        x0 = random.randint(left_start, left_end)
        y0 = 0
        x1 = random.randint(0, width - 1)
        y1 = height - 1
        lines.append(((x0, y0), (x1, y1)))

        x0 = random.randint(right_start, right_end - 1)
        y0 = 0
        x1 = random.randint(0, width - 1)
        y1 = height - 1
        lines.append(((x0, y0), (x1, y1)))

    return lines

# Overlay X-rays on the Image
def overlay_xrays(image, lines):
    """Overlay X-rays (lines) on top of the Male Lena image."""
    overlay = np.copy(image)

    for line in lines:
        (x0, y0), (x1, y1) = line
        rr, cc = bresenham_line(x0, y0, x1, y1)
        for r, c in zip(rr, cc):
            if 0 <= r < overlay.shape[0] and 0 <= c < overlay.shape[1]:
                overlay[r, c] = 1.0  # Brighten the pixel to simulate X-ray

    return overlay

# Bresenham's Line Algorithm
def bresenham_line(x0, y0, x1, y1):
    """Generate pixel coordinates for a line using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    rr, cc = [], []

    while True:
        rr.append(y0)
        cc.append(x0)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return rr, cc

# Main Function to Generate the Image
def main(image_path, num_lines=100):
    """Generate and save the Male Lena image with X-rays overlay."""
    male_lena = load_image(image_path)
    image_shape = male_lena.shape
    lines = generate_xray_lines(image_shape, num_lines)
    overlay_image = overlay_xrays(male_lena, lines)

    # Plotting the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_image, cmap='gray')
    plt.axis('off')
    plt.title("Male Lena with Full X-rays Overlay (Top to Bottom)")
    plt.show()

# Run the main function
if __name__ == "__main__":
    image_path = "Lena_512.png"  # Replace with the path to the Male Lena image
    main(image_path, num_lines=10)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Load Male Lena Image
def load_image(image_path):
    """Load the Male Lena image, convert to grayscale, and normalize."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image, dtype=np.float32) / 255.0

# Generate X-ray Lines from the Very Top
def generate_xray_lines(image_shape, num_lines):
    """Generate X-ray lines originating from the very top edge of the image."""
    height, width = image_shape
    lines = []

    # Divide the image into two vertical regions: left and right halves
    left_start, left_end = 0, width // 2
    right_start, right_end = width // 2, width

    for _ in range(num_lines // 2):
        # Generate random start points at the very top of the image
        x0 = random.randint(left_start, left_end)
        y0 = 0
        x1 = random.randint(0, width - 1)
        y1 = height - 1
        lines.append(((x0, y0), (x1, y1)))

        x0 = random.randint(right_start, right_end - 1)
        y0 = 0
        x1 = random.randint(0, width - 1)
        y1 = height - 1
        lines.append(((x0, y0), (x1, y1)))

    return lines

# Overlay X-rays on the Image
def overlay_xrays(image, lines):
    """Overlay X-rays (lines) on top of the Male Lena image."""
    overlay = np.copy(image)

    for line in lines:
        (x0, y0), (x1, y1) = line
        rr, cc = bresenham_line(x0, y0, x1, y1)
        for r, c in zip(rr, cc):
            if 0 <= r < overlay.shape[0] and 0 <= c < overlay.shape[1]:
                overlay[r, c] = 1.0  # Brighten the pixel to simulate X-ray

    return overlay

# Bresenham's Line Algorithm
def bresenham_line(x0, y0, x1, y1):
    """Generate pixel coordinates for a line using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    rr, cc = [], []

    while True:
        rr.append(y0)
        cc.append(x0)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return rr, cc

# Main Function to Generate the Image
def main(image_path, num_lines=100):
    """Generate and save the Male Lena image with X-rays overlay."""
    male_lena = load_image(image_path)
    image_shape = male_lena.shape
    lines = generate_xray_lines(image_shape, num_lines)
    overlay_image = overlay_xrays(male_lena, lines)

    # Plotting the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_image, cmap='gray')
    plt.axis('off')
    plt.title("Male Lena with Full X-rays Overlay (Top to Bottom)")
    plt.show()

# Run the main function
if __name__ == "__main__":
    image_path = "Lena_512.png"  # Replace with the path to the Male Lena image
    main(image_path, num_lines=100)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Load Male Lena Image
def load_image(image_path):
    """Load the Male Lena image, convert to grayscale, and normalize."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image, dtype=np.float32) / 255.0

# Generate X-ray Lines Covering Full Top
def generate_xray_lines(image_shape, num_lines):
    """Generate X-ray lines starting from the top, including black spaces."""
    height, width = image_shape
    extended_width = width * 2  # Extend the width to include black spaces
    lines = []

    for _ in range(num_lines):
        # Generate random start points across the full extended top
        x0 = random.randint(0, extended_width - 1)
        y0 = 0  # Always start at the very top

        # Generate end points within the actual image dimensions
        x1 = random.randint(0, width - 1)
        y1 = height - 1  # End at the bottom of the image only

        lines.append(((x0, y0), (x1, y1)))

    return lines

# Overlay X-rays on the Image with Black Space
def overlay_xrays(image, lines, extended_width):
    """Overlay X-rays (lines) on top of the Male Lena image with black spaces."""
    # Create a blank image with black spaces
    overlay = np.zeros((image.shape[0], extended_width), dtype=np.float32)

    # Insert the original image in the center
    start_col = (extended_width - image.shape[1]) // 2
    overlay[:, start_col:start_col + image.shape[1]] = image

    for line in lines:
        (x0, y0), (x1, y1) = line
        rr, cc = bresenham_line(x0, y0, x1 + start_col, y1)
        for r, c in zip(rr, cc):
            if 0 <= r < overlay.shape[0] and 0 <= c < overlay.shape[1]:
                overlay[r, c] = 1.0  # Brighten the pixel to simulate X-ray

    return overlay

# Bresenham's Line Algorithm
def bresenham_line(x0, y0, x1, y1):
    """Generate pixel coordinates for a line using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    rr, cc = [], []

    while True:
        rr.append(y0)
        cc.append(x0)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return rr, cc

# Main Function to Generate and Save the Image
def main(image_path, num_lines=100, output_path="output_image.png"):
    """Generate and save the Male Lena image with X-rays overlay."""
    male_lena = load_image(image_path)
    image_shape = male_lena.shape
    extended_width = image_shape[1] * 2  # Double the width to include black spaces
    lines = generate_xray_lines(image_shape, num_lines)
    overlay_image = overlay_xrays(male_lena, lines, extended_width)

    # Save the image using Matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(overlay_image, cmap='gray', extent=(-image_shape[1], 2 * image_shape[1], 0, image_shape[0]))
    plt.axis('off')
    plt.title("Male Lena with Full X-rays Overlay and Black Spaces")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

    # Optionally save the image using Pillow
    overlay_pillow = Image.fromarray((overlay_image * 255).astype(np.uint8))
    overlay_pillow.save("overlay_image_pillow.png")

# Run the main function
if __name__ == "__main__":
    image_path = "Lena_512.png"  # Replace with the path to the Male Lena image
    main(image_path, num_lines=150, output_path="male_lena_xrays.png")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Given tumor sizes (diameters in mm) and corresponding X-ray lines needed
tumor_sizes = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
xray_lines_needed = np.array([1036, 946, 858, 799, 645, 623, 623, 623])

# Create the plot
plt.figure(figsize=(8,5))
plt.plot(tumor_sizes, xray_lines_needed, 'bo-', label="Updated (Monotonic)")

# Labels and title
plt.xlabel("Tumor Diameter (mm)")
plt.ylabel("Number of X-ray Lines Needed")
plt.title("X-ray Line Coverage for Tumors (Adjusted for Monotonicity)")
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig("tumor_diameter_xray_lines_semi_log_monotonic.png")
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from numpy.linalg import norm

# Load Image
def load_image(image_path):
    """Load the image, convert to grayscale, and normalize."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image, dtype=np.float32) / 255.0

# Wu's Line Generation for Simulated Ray Paths
def wu_line(x0, y0, x1, y1, image_shape):
    """Generate Wu's antialiased line pixels."""
    line_pixels = []
    dx = x1 - x0
    dy = y1 - y0
    gradient = abs(dy / dx) if dx != 0 else 1
    y = y0
    ystep = 1 if y1 > y0 else -1

    for x in range(x0, x1 + 1):
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            line_pixels.append((int(y), x))
        y += gradient * ystep
    return line_pixels

# Generate Random Rays
def generate_random_rays(image_shape, num_rays):
    """Generate random rays originating from anywhere around the image."""
    height, width = image_shape
    rays = []
    for _ in range(num_rays):
        x0, y0 = np.random.randint(0, width), np.random.randint(0, height)
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        ray = wu_line(x0, y0, x1, y1, image_shape)
        rays.append(ray)
    return rays

# Generate Fan Beam Rays
def generate_fan_beam_rays(image_shape, num_views=30):
    """Generate fan beam rays from a fixed set of source locations."""
    height, width = image_shape
    rays = []
    top_positions = np.linspace(0, width - 1, num_views, dtype=int)
    bottom_positions = np.arange(0, width)

    for x0 in top_positions:
        for x1 in bottom_positions:
            ray = wu_line(x0, 0, x1, height - 1, image_shape)
            rays.append(ray)
    return rays

# Simulate Projections
def simulate_projections(image, rays):
    """Simulate projections using rays and the input image."""
    return np.array([sum(image[y, x] for y, x in ray) for ray in rays])

# MART Algorithm Implementation
def mart_algorithm(image_shape, rays, projections, iterations=10, learning_rate=0.5):
    """MART reconstruction algorithm."""
    reconstruction = np.ones(image_shape) * 0.5
    epsilon = 1e-6
    for _ in range(iterations):
        for ray, target_sum in zip(rays, projections):
            current_sum = sum(reconstruction[y, x] for y, x in ray) + epsilon
            correction_factor = 1 + learning_rate * (target_sum / current_sum - 1)
            for y, x in ray:
                reconstruction[y, x] *= correction_factor
        reconstruction = gaussian_filter(reconstruction, sigma=1.0)
    return reconstruction / reconstruction.max()

# Run and Plot Reconstructions
def reconstruct_and_plot(image_path, num_random_rays=1000, num_fan_views=30, iterations_list=[50, 100, 200, 500]):
    """Perform reconstruction using both random and fan beam rays and display results with iteration count and norm."""
    original_image = load_image(image_path)
    image_shape = original_image.shape

    fig, axs = plt.subplots(len(iterations_list), 2, figsize=(8, 14))

    for idx, iterations in enumerate(iterations_list):
        # Random Rays Reconstruction
        random_rays = generate_random_rays(image_shape, num_random_rays)
        random_projections = simulate_projections(original_image, random_rays)
        random_reconstruction = mart_algorithm(image_shape, random_rays, random_projections, iterations)
        random_norm = norm(random_reconstruction - original_image)

        # Display Original Image
        axs[idx, 0].imshow(original_image, cmap='gray')
        axs[idx, 0].set_title("Original Image")
        axs[idx, 0].axis('off')

        # Display Reconstructed Image
        axs[idx, 1].imshow(random_reconstruction, cmap='gray')
        axs[idx, 1].set_title(f"Reconstructed (Random Rays)\nIterations: {iterations}, Norm: {random_norm:.2f}")
        axs[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Run Reconstruction
if __name__ == "__main__":
    image_path = 'Lena_512.png'  # Change to your image path
    reconstruct_and_plot(image_path, num_random_rays=5000, num_fan_views=30, iterations_list=[50, 100, 200, 500])


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random

# Generate Perlin-like noise texture (without using the noise module)
def generate_tissue_texture(width, height, sigma, seed):
    np.random.seed(seed)
    texture = np.random.rand(height, width)
    texture = gaussian_filter(texture, sigma=sigma)  # Smooth the noise
    texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255  # Normalize and scale to 0-255
    return texture

# Add numbered disks to texture
def add_disks_to_texture(texture, num_disks=100, disk_diameter=30, max_density=255):
    disk_radius = disk_diameter // 2
    added_disks = 0
    densities = np.linspace(0.01 * max_density, max_density, num_disks)
    disk_centers = []
    
    while added_disks < num_disks:
        x, y = random.randint(disk_radius, texture.shape[1] - disk_radius), random.randint(disk_radius, texture.shape[0] - disk_radius)
        
        # Ensure no overlap with existing disks
        if all((np.hypot(x - xc, y - yc) > disk_diameter) for xc, yc, _ in disk_centers):
            disk_centers.append((x, y, densities[added_disks]))
            
            # Fill in disk density
            for dx in range(-disk_radius, disk_radius + 1):
                for dy in range(-disk_radius, disk_radius + 1):
                    if dx**2 + dy**2 <= disk_radius**2 and 0 <= x + dx < texture.shape[1] and 0 <= y + dy < texture.shape[0]:
                        texture[y + dy, x + dx] = densities[added_disks]
            
            added_disks += 1
    
    return texture, disk_centers

# Visualize with labeled disks
def visualize_texture_with_disks(texture, disk_centers, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(texture, cmap="gray")
    for x, y, density in disk_centers:
        color = "black" if density > 0.5 * 255 else "white"  # Black for dark disks, white for light disks
        plt.text(x, y, f"{int(density)}", color=color, ha='center', va='center', fontsize=8)
    plt.title(title)
    plt.colorbar(label='Density')
    plt.axis("off")
    plt.show()

# Main function to generate and visualize Perlin noise maps for different tissue types
def main():
    width, height = 512, 512

    # Generate Perlin-like textures for different tissue types
    adipose_texture = generate_tissue_texture(width, height, sigma=20, seed=42)  # Adipose: smooth
    glandular_texture = generate_tissue_texture(width, height, sigma=10, seed=99)  # Glandular: intricate
    mixed_texture = generate_tissue_texture(width, height, sigma=15, seed=77)  # Mixed: intermediate

    # Add disks and visualize each texture
    for texture, tissue_name in zip([adipose_texture, glandular_texture, mixed_texture],
                                    ["Adipose", "Glandular", "Mixed"]):
        texture_with_disks, disk_centers = add_disks_to_texture(texture.copy(), disk_diameter=30)
        visualize_texture_with_disks(texture_with_disks, disk_centers, f"{tissue_name} Tissue with Density-Labeled Disks")

if __name__ == "__main__":
    main()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Generate Perlin-like noise using random arrays and Gaussian filtering
def generate_perlin_like_noise(shape=(512, 512), scale=50, seed=0):
    """
    Generates Perlin-like noise using random arrays and Gaussian smoothing.
    Parameters:
        shape (tuple): Shape of the output array.
        scale (int): Controls the smoothness of the noise.
        seed (int): Random seed for reproducibility.
    Returns:
        np.ndarray: A 2D array of Perlin-like noise values.
    """
    np.random.seed(seed)
    noise = np.random.rand(*shape)  # Random noise
    noise = gaussian_filter(noise, sigma=scale)  # Smooth the noise
    return noise / np.max(noise)  # Normalize between 0 and 1

# Plot Perlin-like noise
def plot_perlin_like_noise(noise, title, save_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(noise, cmap='gray', interpolation='nearest')  # Grayscale image
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Main function
def main():
    shape = (512, 512)  # Shape of the noise map

    # Adipose tissue parameters
    adipose_noise = generate_perlin_like_noise(shape, scale=20, seed=42)
    plot_perlin_like_noise(adipose_noise, "Adipose Tissue Perlin-Like Noise", "adipose_perlin_like_noise.png")

    # Glandular tissue parameters
    glandular_noise = generate_perlin_like_noise(shape, scale=10, seed=99)
    plot_perlin_like_noise(glandular_noise, "Glandular Tissue Perlin-Like Noise", "glandular_perlin_like_noise.png")

    # Mixed tissue parameters
    mixed_noise = generate_perlin_like_noise(shape, scale=15, seed=77)
    plot_perlin_like_noise(mixed_noise, "Mixed Tissue Perlin-Like Noise", "mixed_perlin_like_noise.png")

if __name__ == "__main__":
    main()


# In[ ]:


import matplotlib.pyplot as plt
import networkx as nx
import random

# Define the X-ray lines with one isolated and others forming a connected group
def generate_top_to_bottom_lines(num_lines=6, width=10, height=10):
    lines = []
    
    # Ensure one line is isolated (does not intersect with others)
    isolated_line = [(random.uniform(0, width / 4), height), (random.uniform(0, width / 4), 0)]
    lines.append(isolated_line)
    
    # Generate lines that form a connected group
    start_x = width / 2  # Start group lines roughly from the center
    for _ in range(num_lines - 1):
        x1, y1 = random.uniform(start_x - 2, start_x + 2), height  # Random x near the center at the top
        x2, y2 = random.uniform(start_x - 2, start_x + 2), 0       # Random x near the center at the bottom
        lines.append([(x1, y1), (x2, y2)])
    
    return lines

# Create dual graph for the lines
def create_dual_graph(lines):
    G = nx.Graph()
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i < j:  # Avoid duplicate edges
                x1, y1, x2, y2 = *line1[0], *line1[1]
                x3, y3, x4, y4 = *line2[0], *line2[1]
                # Simplified logic: Check if projections on x-axis overlap
                if not (max(x1, x2) < min(x3, x4) or max(x3, x4) < min(x1, x2)):
                    G.add_edge(i + 1, j + 1)
    return G

# Plot the X-ray lines in real space
def plot_real_space(lines):
    plt.figure(figsize=(8, 8))
    for i, line in enumerate(lines):
        x_coords, y_coords = zip(*line)
        plt.plot(x_coords, y_coords, label=f'Line {i + 1}')
        midpoint = ((x_coords[0] + x_coords[1]) / 2, (y_coords[0] + y_coords[1]) / 2)
        plt.text(midpoint[0], midpoint[1], f'{i + 1}', color='red', fontsize=12)
    plt.title("Real Space Representation of X-ray Lines")
    # Remove axis numbers and gridlines
    plt.xticks([])  # Remove x-axis numbers
    plt.yticks([])  # Remove y-axis numbers
    plt.grid(False)  # Remove gridlines
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()


# Plot the dual graph
def plot_dual_graph(G):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # Layout for graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10)
    plt.title("Dual Graph Representation of X-ray Line Intersections")

# Main function
def main():
    lines = generate_top_to_bottom_lines(num_lines=6, width=10, height=10)
    G = create_dual_graph(lines)
    
    # Plot real space
    plot_real_space(lines)
    plt.show()
    
    # Plot dual graph
    plot_dual_graph(G)
    plt.show()

if __name__ == "__main__":
    main()

