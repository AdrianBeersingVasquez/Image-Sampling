import matplotlib.image as mpimg
from image_sampling import *

def main():
    random.seed(0)
    
    # Read image
    path_to_file = 'london1.jpg'
    img = mpimg.imread(path_to_file)

    # Get number and size of samples
    num_samples = 3
    sample_length = 141#get_sample_length()

    # Check if the image dimensions are big enough to fit the samples
    orientations = get_valid_orientations(img, sample_length)
    assert len(orientations) > 0, f'The sample sizes are too large. {num_samples} samples of length {sample_length} cannot fit within the image without overlap'
    
    # Select inital sample layout and calculate inital sample coordinates
    selected_orientation = orientations[0]
    sample_coordinates = initialise_sample_coordinates(selected_orientation, sample_length, num_samples)
    
    # (Randomly) shuffle position of samples
    for i in range(num_samples):
        # Refresh the position map
        position_map = create_position_map(img)
        
        # Place all samples (except for the sample to be moved)
        for j in range(num_samples):
            if i != j:
                position_map = place_sample_on_map(position_map, sample_coordinates[j], sample_length)
        
        # Add a buffer around each placed sample (prevent overlap with other samples)
        # and add border, if necessary (to contain sample fully within image)
        position_map = dilate_zeros(position_map, sample_length)
        if sample_length>1:
            position_map = add_border(position_map, halve_length(sample_length))
        
        # Select new postion from available spots. Update with new coordinates 
        new_pos = randomly_place_sample(position_map)
        sample_coordinates[i] = new_pos
    
    # Save sampled images
    samples = []
    for i in range(num_samples):
        sample = sample_image(img, sample_coordinates[i], sample_length//2)
        samples.append(sample)
    
    plot_samples(img, num_samples, sample_coordinates, halve_length(sample_length, decimalise=True), halve_length(sample_length), samples)

if __name__ == '__main__':
    main()
