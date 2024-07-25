import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def main():
    #random.seed(31)
    
    # Read image
    path_to_file = 'london1.jpg'
    img = mpimg.imread(path_to_file)
    #plt.imshow(img)
    #plt.show()
    
    # Get number and size of samples
    num_samples = 3
    sample_size = 21
    sample_length = sample_size
    half_sample_length = sample_length
    half_size = half_sample_length


    
    # Check if the image dimensions are big enough to fit the samples
    orientations = get_valid_orientations(img, num_samples, sample_size)
    if len(orientations) == 0:
        print(f'The sample sizes are too large. {num_samples} samples of length {sample_size} cannot fit within the image without overlap')
        return
    
    position_map = create_position_map(img)
    
    # Select inital sample layout and calculate inital sample coordinates
    selected_orientation = orientations[0]
    sample_coordinates = initialise_sample_coordinates(selected_orientation, sample_size, num_samples)
    print('Sample coordinates:', sample_coordinates)
    
    # Place samples on position_map (setting corresponding values to 0)
    for i in range(len(sample_coordinates)):
        print(sample_coordinates[i])
        position_map = place_sample_on_map(position_map, sample_coordinates[i], sample_size)
    
    for i in range(num_samples):
        # Refresh the position map
        position_map = create_position_map(img)
        
        # Place all samples (except for the sample to be moved)
        for j in range(num_samples):
            if i != j:
                position_map = place_sample_on_map(position_map, sample_coordinates[j], sample_size)
        
        # Add a buffer around each placed sample (prevent overlap with other samples)
        # and add border (to contain sample fully within image)
        position_map = dilate_zeros(position_map, sample_size)
        position_map = add_border(position_map, sample_size//2)
        
        # Select new postion from available spots. Update with new coordinates 
        new_pos = randomly_place_sample(position_map, sample_size)
        sample_coordinates[i] = new_pos
    
    # Save sampled images
    samples = []
    for i in range(num_samples):
        sample = sample_image(img, sample_coordinates[i], sample_size//2)
        samples.append(sample)
    
    # Plot all samples in a single row below the original image
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].imshow(img)
    for i in range(num_samples):
        x, y = sample_coordinates[i]
        rect = plt.Rectangle((x - half_size, y - half_size), sample_size, sample_size, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)
    axs[0].axis('off')
    
    # Display the samples in a single row
    axs[1].imshow(np.hstack(samples))
    axs[1].axis('off')
    
    plt.show()

if __name__ == '__main__':
    main()
