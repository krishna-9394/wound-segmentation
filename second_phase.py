import cv2
import numpy as np
import os

def mark_tissues(input_image_path, segmented_image_path, output_image_path):
    # Load the original input image and the segmented wound image
    input_image = cv2.imread(input_image_path)
    segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the segmented image to match the input image dimensions, if necessary
    if input_image.shape[:2] != segmented_image.shape[:2]:
        segmented_image = cv2.resize(segmented_image, (input_image.shape[1], input_image.shape[0]))
    
    # Define refined intensity ranges for tissue classification
    granulation_range = (200, 255)  # Bright red
    necrotic_range = (100, 199)     # Dark brown/black
    slough_range = (30, 99)         # Pale yellow to brownish-yellow
    
    # Create a blank color image for tissue annotation
    tissue_marked_image = np.zeros_like(input_image)
    
    # Apply color coding based on intensity ranges
    tissue_marked_image[(segmented_image >= granulation_range[0]) & (segmented_image <= granulation_range[1])] = [0, 0, 255]  # Red for Granulation
    tissue_marked_image[(segmented_image >= necrotic_range[0]) & (segmented_image <= necrotic_range[1])] = [0, 255, 255]  # Yellow for Necrotic
    tissue_marked_image[(segmented_image >= slough_range[0]) & (segmented_image <= slough_range[1])] = [0, 0, 0]  # Black for Slough

    # Combine the marked image with the original input image
    # Use weighted addition to retain visibility of the underlying image
    blended_image = cv2.addWeighted(input_image, 0.5, tissue_marked_image, 0.5, 0)
    
    # Save the output image
    cv2.imwrite(output_image_path, blended_image)

# Paths to the directories
input_image_dir = 'data/images/'
segmented_image_dir = 'predictions/phase_1/'
output_image_dir = 'predictions/phase_2/'

# Ensure the output directory exists
os.makedirs(output_image_dir, exist_ok=True)

# Process images from 1.png to 27.png
for i in range(1, 28):  # Loop from 1 to 27
    input_image_path = os.path.join(input_image_dir, f'{i}.png')
    segmented_image_path = os.path.join(segmented_image_dir, f'{i}.png')
    output_image_path = os.path.join(output_image_dir, f'{i}.png')
    
    try:
        mark_tissues(input_image_path, segmented_image_path, output_image_path)
        print(f"Processed image {i}.png successfully.")
    except Exception as e:
        print(f"Failed to process image {i}.png: {e}")
