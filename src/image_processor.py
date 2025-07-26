import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2 # Import OpenCV

# --- 1. Load Image ---
image_path = "my_image.jpg"  # Change this to your image file name
# Ensure your image is in the same directory as this script, or provide the full path.
image_array = None
try:
    # Use PIL to open, then convert to NumPy array.
    image_pil = Image.open(image_path)
    image_array = np.array(image_pil)
    print(f"Image loaded: {image_path}")
    print(f"Original Image shape: {image_array.shape}")
    print(f"Original Image Data type: {image_array.dtype}")
except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found.")
    print("Please ensure the image file is in the same directory or the path is correct.")
    exit() # Exit the script if image is not found
except Exception as e:
    print(f"An unexpected error occurred during image loading: {e}")
    exit()

# --- 2. Convert to Grayscale (NumPy method - Needed early for histogram) ---
gray_image_array = None
if len(image_array.shape) == 3 and image_array.shape[2] == 3:
    gray_image_array = (
        0.2989 * image_array[:, :, 0].astype(np.float32) +
        0.5870 * image_array[:, :, 1].astype(np.float32) +
        0.1140 * image_array[:, :, 2].astype(np.float32)
    ).astype(np.uint8)
    print("Converted to grayscale (NumPy method).")
else:
    gray_image_array = image_array.copy()
    print("Image is already grayscale.")

# --- 3. Initial Image Overview: Original, Grayscale, and Histogram --- (NEW COMBINED SECTION)
plt.figure(figsize=(15, 6)) # Adjust figure size to fit 3 plots horizontally

# Plot 1: Original Image
plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
if len(image_array.shape) == 3 and image_array.shape[2] == 3:
    plt.imshow(image_array)
    plt.title("1. Original Image (Color)")
else:
    plt.imshow(image_array, cmap='gray')
    plt.title("1. Original Image (Grayscale)")
plt.axis('off')

# Plot 2: Grayscale Image
plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd plot
plt.imshow(gray_image_array, cmap='gray')
plt.title("2. Grayscale Image (NumPy)")
plt.axis('off')

# Plot 3: Histogram of Grayscale Pixel Intensities
plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd plot
gray_for_stats = gray_image_array.astype(np.float32) / 255.0 # Ensure 0-1 range for histogram
plt.hist(gray_for_stats.ravel(), bins=256, range=(0.0, 1.0), density=True, color='gray', alpha=0.7)
plt.title("3. Histogram of Pixel Intensities")
plt.xlabel("Pixel Intensity (0.0 = Black, 1.0 = White)")
plt.ylabel("Normalized Frequency")
plt.grid(axis='y', alpha=0.75)

plt.tight_layout() # Adjust spacing between subplots
plt.show() # Display this combined initial figure

print("\n--- Grayscale Image Statistics ---")
print(f"Min Pixel Value: {np.min(gray_for_stats):.4f}")
print(f"Max Pixel Value: {np.max(gray_for_stats):.4f}")
print(f"Mean Pixel Value: {np.mean(gray_for_stats):.4f}")
print(f"Standard Deviation of Pixel Values: {np.std(gray_for_stats):.4f}")


# --- 4. Basic Image Processing Operations (NumPy implementations) ---

# 4.1. Crop the Image (center crop)
cropped_image_array = None
height, width = gray_image_array.shape
crop_size = min(height, width) // 2
start_row = (height - crop_size) // 2
start_col = (width - crop_size) // 2
cropped_image_array = gray_image_array[start_row:start_row+crop_size, start_col:start_col+crop_size]
print(f"\nCropped image to size: {cropped_image_array.shape} (NumPy)")

# 4.2. Simple Edge Detection Filter (Sobel-like for X-direction - NumPy)
edge_kernel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
gray_float = gray_image_array.astype(np.float32)
height, width = gray_float.shape
np_edge_detected_image_array = np.zeros((height - 2, width - 2), dtype=np.float32)

for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = gray_float[i-1:i+2, j-1:j+2]
        np_edge_detected_image_array[i-1, j-1] = np.sum(neighborhood * edge_kernel_x)

np_edge_detected_image_array = np.abs(np_edge_detected_image_array)
if np.max(np_edge_detected_image_array) > 0:
    np_edge_detected_image_array = np_edge_detected_image_array / np.max(np_edge_detected_image_array)
np_edge_detected_image_array = (np_edge_detected_image_array * 255).astype(np.uint8)

print(f"Edges detected (NumPy). Dimensions: {np_edge_detected_image_array.shape}")

# 4.3. Simple Blur Filter (3x3 mean filter - NumPy)
gray_float_for_blur = gray_image_array.astype(np.float32)
height, width = gray_float_for_blur.shape
np_blurred_image_array = np.zeros((height - 2, width - 2), dtype=np.float32)

for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = gray_float_for_blur[i-1:i+2, j-1:j+2]
        np_blurred_image_array[i-1, j-1] = np.mean(neighborhood)

np_blurred_image_array = np_blurred_image_array.astype(np.uint8)

print(f"Blurred image created (NumPy). Dimensions: {np_blurred_image_array.shape}")


# --- 5. Use Pandas for Image Metadata Management ---
metadata = {
    'File Name': [image_path],
    'Original Dimensions': [f"{image_array.shape[0]}x{image_array.shape[1]}"],
    'Is Color': [len(image_array.shape) == 3 and image_array.shape[2] == 3],
    'Has Grayscale': [gray_image_array is not None],
    'Has Cropped': [cropped_image_array is not None],
    'Has NumPy Blurred': [np_blurred_image_array is not None],
    'Has NumPy Edge Detected': [np_edge_detected_image_array is not None],
    'Notes': ["Basic image processing demo using NumPy, Pandas, PIL, Matplotlib, and OpenCV with advanced features"]
}
df = pd.DataFrame(metadata)
print("\n--- Image Metadata ---")
print(df.to_string(index=False))


# --- 6. Introducing OpenCV for Basic Image Processing ---
# Convert original image_array (RGB from PIL) to BGR for OpenCV if it's color
cv_image_array_bgr = image_array
if len(image_array.shape) == 3 and image_array.shape[2] == 3:
    cv_image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

# Always use the grayscale version for filtering for simplicity
cv_gray_image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
print(f"\nGrayscale conversion (OpenCV). Dimensions: {cv_gray_image_array.shape}")

# 6.2. Blurring with OpenCV (Gaussian Blur)
cv_blurred_image_array = cv2.GaussianBlur(cv_gray_image_array, (5, 5), 0)
print(f"Blurred image created (OpenCV Gaussian Blur). Dimensions: {cv_blurred_image_array.shape}")

# 6.3. Edge Detection with OpenCV (Canny Edge Detector)
cv_canny_edges_image_array = cv2.Canny(cv_gray_image_array, 100, 200) # Adjust thresholds as needed
print(f"Edges detected (OpenCV Canny). Dimensions: {cv_canny_edges_image_array.shape}")


# --- 7. Advanced OpenCV Operations ---

# 7.1. Image Thresholding (Binary Conversion)
ret, thresholded_image = cv2.threshold(cv_gray_image_array, 127, 255, cv2.THRESH_BINARY)
print(f"\nImage Thresholded. Dimensions: {thresholded_image.shape}")

# 7.2. Morphological Operations (Erosion and Dilation)
kernel = np.ones((5,5), np.uint8)

eroded_image = cv2.erode(thresholded_image, kernel, iterations = 1)
print(f"Image Eroded. Dimensions: {eroded_image.shape}")

dilated_image = cv2.dilate(thresholded_image, kernel, iterations = 1)
print(f"Image Dilated. Dimensions: {dilated_image.shape}")

# 7.3. Contour Detection
contours, hierarchy = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = np.zeros_like(cv_image_array_bgr) # Use color image size for color contours
cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)
print(f"Detected {len(contours)} contours.")


# --- 8. Consolidated Display of All Processed Images (NumPy & OpenCV Advanced) ---
# This will display 3 rows x 4 columns = 12 plots
plt.figure(figsize=(24, 18)) # Increased figure size for more plots

# Row 1: NumPy processed
plt.subplot(3, 4, 1) # (rows, columns, plot_number)
plt.imshow(cropped_image_array, cmap='gray')
plt.title("Cropped (NumPy)")
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(np_blurred_image_array, cmap='gray')
plt.title("Blurred (NumPy Mean)")
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(np_edge_detected_image_array, cmap='gray')
plt.title("Edges (NumPy Sobel X)")
plt.axis('off')

# Row 2: Basic OpenCV processed
plt.subplot(3, 4, 5) # Note: plot 4 is empty, intentionally skipped for better grouping
plt.imshow(cv_gray_image_array, cmap='gray')
plt.title("Grayscale (OpenCV)")
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(cv_blurred_image_array, cmap='gray')
plt.title("Blurred (OpenCV Gaussian)")
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(cv_canny_edges_image_array, cmap='gray')
plt.title("Edges (OpenCV Canny)")
plt.axis('off')

# Row 3: Advanced OpenCV
plt.subplot(3, 4, 9)
plt.imshow(thresholded_image, cmap='gray')
plt.title("Thresholded (OpenCV)")
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(eroded_image, cmap='gray')
plt.title("Eroded (OpenCV)")
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(dilated_image, cmap='gray')
plt.title("Dilated (OpenCV)")
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)) # Convert back to RGB for Matplotlib
plt.title(f"Detected Contours ({len(contours)})")
plt.axis('off')

# Add a placeholder for a blank subplot or a text description
plt.subplot(3, 4, 4)
plt.axis('off') # Turn off axes for the empty plot
plt.text(0.5, 0.5, "NumPy Custom Filters", 
         horizontalalignment='center', verticalalignment='center',
         fontsize=14, color='darkgray')

plt.subplot(3, 4, 8)
plt.axis('off') # Turn off axes for the empty plot
plt.text(0.5, 0.5, "OpenCV Basic Filters", 
         horizontalalignment='center', verticalalignment='center',
         fontsize=14, color='darkgray')


plt.tight_layout()
plt.show() # This final plt.show() displays the consolidated grid