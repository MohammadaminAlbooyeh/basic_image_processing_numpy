import matplotlib.pyplot as plt
import numpy as np

# Set the path to your image file here
# Make sure the file name and extension are correct and the file is in the same directory
image_path = 'my_image.jpg' # Replace with your actual image file name

try:
    # Read the image using Matplotlib
    # Matplotlib reads the image as a NumPy array
    image_array = plt.imread(image_path)

    print(f"Image successfully loaded. Image dimensions: {image_array.shape}")
    print(f"Data type of the array: {image_array.dtype}")

    # --- Convert image to grayscale ---
    gray_image_array = None # Initialize to None in case conversion fails or isn't needed

    # Check if the image is colored (has 3 RGB channels)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Common formula for grayscale conversion (ITU-R BT.601 standard)
        # This formula assigns different weights to R, G, B channels because the human eye
        # is more sensitive to green.
        gray_image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
        print(f"Image converted to grayscale. New dimensions: {gray_image_array.shape}")
        print(f"Data type of the grayscale array: {gray_image_array.dtype}")
    elif len(image_array.shape) == 2:
        print("Image is already grayscale (single channel). No conversion needed.")
        gray_image_array = image_array # If it's already grayscale, use it as is
    else:
        print("Image format not supported for grayscale conversion or unexpected dimensions.")
        # gray_image_array remains None, or you could assign original for display purposes.

    # --- Display images ---
    plt.figure(figsize=(10, 5)) # Create a new figure to display multiple plots side-by-side

    # Plot original image
    plt.subplot(1, 2, 1) # Specify plot location (1 row, 2 columns, 1st plot)
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.axis('off')

    # Plot grayscale image (only if conversion was successful)
    if gray_image_array is not None:
        plt.subplot(1, 2, 2) # Specify plot location (1 row, 2 columns, 2nd plot)
        plt.imshow(gray_image_array, cmap='gray') # 'cmap='gray' is crucial for correct grayscale display
        plt.title("Grayscale Image")
        plt.axis('off')

    plt.tight_layout() # Adjust spacing between subplots for better appearance
    plt.show()

except FileNotFoundError:
    print(f"Error: File '{image_path}' not found. Please ensure the path and file name are correct.")
except Exception as e:
    print(f"An error occurred: {e}")