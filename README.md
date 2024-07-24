# Image Sampler

## Overview

This Image Sampler script is designed to extract 3 non-overlapping random samples from a given image. The user defines the size of the three samples, and if the image is large enough to fit three non-overlapping samples, the images are displayed.

## Approach

When executed, the script will:

1. Read and display the input image.

2. Based on the dimensions of the image, verify that the samples can be arranged over the input image, without sample overlap.

3. The samples are placed on the image in a valid configuration, as determined by step 2.

4. A mask is created, by removing the coordinates where samples are placed, including a buffer around the samples and the edge of the image.

5. An image is "lifted", and the mask is used to determine a random valid coordinate where the image can be placed. 

6. Step 5 is repeated for the remaining image samples.

7. The samples are now displayed, and the sample coordinates are saved.


