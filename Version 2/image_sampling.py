import numpy as np
import matplotlib.pyplot as plt
import random

def get_valid_orientations(img, n, sample_size):
    """Determine valid arrangement of samples that can fit within image dimensions """
    img_height, img_width = img.shape[:2]
    valid_orientations = []
    all_orientations = np.array([[3, 1], [2, 2], [1, 3]])
    # Save each arrangement of samples that can fit within the image's dimensions
    for orientation in all_orientations:
        if orientation[0] * sample_size <= img_width and orientation[1] * sample_size <= img_height:
            valid_orientations.append(orientation)
    return np.array(valid_orientations)

def create_position_map(img):
    """Create a map to store valid locations for samples"""
    img_height, img_width = img.shape[:2]
    position_map = np.ones([img_height, img_width])
    return position_map

def add_border(position_map, border_size):
    """Add a border of 0s around the edge of the positional map"""
    position_map[:border_size, :] = 0
    position_map[-border_size:, :] = 0
    position_map[:, :border_size] = 0
    position_map[:, -border_size:] = 0
    return position_map

def initialise_sample_coordinates(orientation, sample_size, num_samples):
    """Calculate and store intial 3 coordinates based on chosen orientation. Return as dict """
    border = sample_size//2
    rows, cols = orientation
    sample_coordinates = {}
    samples_placed = 0
    for row in range(rows):
        for col in range(cols):
            if samples_placed >= num_samples:
                return sample_coordinates
            x = row * sample_size + border
            y = col * sample_size + border
            sample_coordinates[samples_placed] = (x, y)
            samples_placed += 1
    return sample_coordinates

def place_sample_on_map(position_map, coordinates, sample_size):
    """Mark a sample on the positional map by setting the area aroudn the coordinates (the sample region) to 0"""
    half_size = sample_size//2
    x, y = coordinates
    updated_map = position_map
    
    updated_map[y-half_size:y+half_size+1, x-half_size:x+half_size+1] = 0
    
    return updated_map

def dilate_zeros(position_map, sample_size):
    """Dilate all regions where 0 is present by half of sample_size """
    dilated_map = position_map.copy()
    half_size = sample_size // 2
    img_height, img_width = dilated_map.shape
    
    for y in range(img_height):
        for x in range(img_width):
            if position_map[y, x] == 0:
                y_start = max(0, y - half_size)
                y_end = min(img_height, y + half_size + 1)
                x_start = max(0, x - half_size)
                x_end = min(img_width, x + half_size + 1)
                dilated_map[y_start:y_end, x_start:x_end] = 0
    return dilated_map

def randomly_place_sample(position_map, sample_size):
    """Use the positional map as a mask to determine valid spaces where the sample can be (randomly) placed"""
    img_height, img_width = position_map.shape
    
    # Create an array of coordinates equal in shape to original image
    coords = np.array([(x, y) for y in range(img_height) for x in range(img_width)])
    
    # Mask out the invalid positions
    valid_coords = coords[position_map.flatten() == 1]
    
    random_pos = random.choice(valid_coords)
    
    return random_pos

def sample_image(img, coordinates, half_sample_size):
    """Return image centred at given coordinate, of size sample_size"""
    x, y = coordinates
    return img[y-half_sample_size:y+half_sample_size, x-half_sample_size:x+half_sample_size]

def plot_samples(img, num_samples, sample_coordinates, offset, square_size, samples):
    """Plot original image, showing where samples are taken from, and image samples"""
    # Plot original image, overlaying squares to show where samples are taken from
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].imshow(img)
    for i in range(num_samples):
        x, y = sample_coordinates[i]
        rect = plt.Rectangle((x - offset, y -  offset), square_size, square_size, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)
    axs[0].axis('off')
    
    # Display the samples in a single row beneath the original image
    axs[1].imshow(np.hstack(samples))
    axs[1].axis('off')

    plt.show()
