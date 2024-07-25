import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from image_sampling import *

def main():
    #random.seed(0)
    
    # Read image
    path_to_file = 'london1.jpg'
    img = mpimg.imread(path_to_file)
    
    # Get number and size of samples
    num_samples = 3
    sample_size = 81
    sample_length = sample_size
    half_sample_length = sample_length // 2
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
    
    plot_samples(img, num_samples, sample_coordinates, half_sample_length, sample_length, samples)

if __name__ == '__main__':
    main()
