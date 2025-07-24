import matplotlib.pyplot as plt
import numpy as np # Although not directly used here, it's essential for later steps

# Set the path to your image file here
# Make sure the file name and extension are correct and the file is in the same directory
image_path = 'my_image.jpg' # Replace with your actual image file name

try:
    # Read the image using Matplotlib
    # Matplotlib reads the image as a NumPy array
    image_array = plt.imread(image_path)

    print(f"Image successfully loaded. Image dimensions: {image_array.shape}")
    print(f"Data type of the array: {image_array.dtype}")

    # Display the image
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.axis('off') # To hide the axes
    plt.show()

except FileNotFoundError:
    print(f"Error: File '{image_path}' not found. Please ensure the path and file name are correct.")
except Exception as e:
    print(f"An error occurred: {e}")