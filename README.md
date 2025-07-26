# Basic Image Processing with NumPy, OpenCV, and Matplotlib

This project demonstrates basic and advanced image processing techniques using Python libraries: **NumPy**, **OpenCV**, **Matplotlib**, **Pandas**, and **Pillow (PIL)**.  
It covers both custom (NumPy-based) and OpenCV-based operations, and visualizes the results in a consolidated grid.

---

## Features

- **Image Loading**: Loads an image using PIL and converts it to a NumPy array.
- **Grayscale Conversion**: Converts color images to grayscale using NumPy or OpenCV.
- **Visualization**: Shows the original, grayscale, and histogram of pixel intensities.
- **NumPy Operations**:
  - Center cropping
  - Custom edge detection (Sobel X)
  - Custom blurring (mean filter)
- **OpenCV Operations**:
  - Grayscale conversion
  - Gaussian blur
  - Canny edge detection
  - Thresholding (binary conversion)
  - Morphological operations (erosion, dilation)
  - Contour detection and drawing
- **Metadata Table**: Uses Pandas to display image metadata and processing steps.
- **Consolidated Visualization**: Displays all processed images in a 3x4 grid using Matplotlib.

---

## Requirements

- Python 3.7+
- numpy
- matplotlib
- pandas
- pillow
- opencv-python

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Place your image** (e.g., `my_image.jpg`) in the same directory as `image_processor.py`, or update the `image_path` variable in the script to your image's path.

2. **Run the script**:

```bash
python src/image_processor.py
```

3. **View the output**:
   - The script prints image statistics and metadata to the terminal.
   - Multiple Matplotlib windows will display the original, processed, and filtered images in a grid.

---

## File Structure

```
basic_image_processing_numpy/
├── requirements.txt
└── src/
    └── image_processor.py
```

---

## Example Output

- **Terminal**:  
  - Image shape, data type, grayscale statistics, and metadata table.
- **Matplotlib Windows**:  
  - Original, grayscale, histogram, cropped, blurred, edge-detected, thresholded, eroded, dilated, and contour images.

---

## Notes

- The script is designed for educational and demonstration purposes.
- You can easily extend it with more filters or processing steps.
- Make sure your image file is not too large for quick processing and display.

---

## License

This project is provided for educational use.  
Feel free to modify and use it in your own