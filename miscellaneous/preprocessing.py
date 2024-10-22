import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded properly
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    else:
        print(f"Image loaded successfully from {image_path}")
    
    # Resize the image to 256x256 pixels
    image_resized = cv2.resize(image, (256, 256))
    
    # Normalize the image (scale pixel values to [0, 1])
    image_normalized = image_resized / 255.0
    
    # Apply Gaussian Blur to reduce noise (optional)
    image_denoised = cv2.GaussianBlur(image_normalized, (5, 5), 0)
    
    return image_normalized

# Directory paths
input_dir = r'C:\Users\vikra\wound_segmentation_env\wound_segmentation_project\data\images'
output_dir = r'C:\Users\vikra\wound_segmentation_env\wound_segmentation_project\data\preprocessed_images'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a JPEG image
    if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
        image_path = os.path.join(input_dir, filename)
        
        # Print the full path for debugging
        print(f"Trying to load image from: {image_path}")
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        if preprocessed_image is not None:
            # Convert the normalized image back to uint8 format (0-255) for saving
            image_to_save = (preprocessed_image * 255).astype(np.uint8)
            
            # Full path to save the preprocessed image
            save_path = os.path.join(output_dir, filename)
            
            # Save the image
            cv2.imwrite(save_path, image_to_save)
            print(f"Preprocessed image saved to {save_path}")
        else:
            print(f"Skipping image {filename} due to loading error.")
