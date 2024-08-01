import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from image_sampling import *

def halve_length(length):
    """Handle case where sample size is """
    #if length == 1:
    #    return 1
    
    return length//2

def get_sample_length() -> int:
    while True:
        try:
            sample_length = int(input('Size of samples (enter a positive odd integer): '))
            
            if sample_length <= 0:
                print('Invalid input. Sample length must have positive value.')
                continue
            
            if sample_length % 2 == 0:
                print('Invalid input. Sample length must be odd.')
                continue
            
            return sample_length
        
        except ValueError:
            print('Invalid input. Sample length must be an integer.')

def main():
    random.seed(0)
    
    # Read image
    path_to_file = 'london1.jpg'
    img = mpimg.imread(path_to_file)
    
    # Get number and size of samples
    num_samples = 3
    sample_length = get_sample_length()

    # Check if the image dimensions are big enough to fit the samples
    orientations = get_valid_orientations(img, sample_length)
    assert len(orientations) > 0, f'The sample sizes are too large. {num_samples} samples of length {sample_length} cannot fit within the image without overlap'
    
    position_map = create_position_map(img)
    
    # Select inital sample layout and calculate inital sample coordinates
    selected_orientation = orientations[0]
    sample_coordinates = initialise_sample_coordinates(selected_orientation, sample_length, num_samples)
    print('Sample coordinates:', sample_coordinates)
    
    # Place samples on position_map (setting corresponding values to 0)
    for i in range(len(sample_coordinates)):
        print(sample_coordinates[i])
        position_map = place_sample_on_map(position_map, sample_coordinates[i], sample_length)
    
    for i in range(num_samples):
        # Refresh the position map
        position_map = create_position_map(img)
        
        # Place all samples (except for the sample to be moved)
        for j in range(num_samples):
            if i != j:
                position_map = place_sample_on_map(position_map, sample_coordinates[j], sample_length)
        
        # Add a buffer around each placed sample (prevent overlap with other samples)
        # and add border (to contain sample fully within image)
        position_map = dilate_zeros(position_map, sample_length)
        position_map = add_border(position_map, halve_length(sample_length))
        
        # Select new postion from available spots. Update with new coordinates 
        new_pos = randomly_place_sample(position_map)
        sample_coordinates[i] = new_pos
    
    # Save sampled images
    samples = []
    for i in range(num_samples):
        sample = sample_image(img, sample_coordinates[i], halve_length(sample_length))
        samples.append(sample)
    
    plot_samples(img, num_samples, sample_coordinates, halve_length(sample_length), sample_length, samples)

if __name__ == '__main__':
    main()
