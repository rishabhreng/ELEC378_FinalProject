from __future__ import print_function
import cv2 as cv
import numpy as np

# Load the same test image
src = cv.imread(cv.samples.findFile('./test_images/test_images/test_000998.jpg'))

# Show source image
cv.imshow('Source Image', src)

# Remove white background
src[np.all(src == 255, axis=2)] = 0

# Convert to grayscale for edge detection
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv.GaussianBlur(gray, (5, 5), 1.5)

# Apply Canny edge detection
edges = cv.Canny(blurred, 100, 150)

cv.imshow('Canny Edges', edges)

# Optional: Apply morphological operations to enhance edges
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
edges_dilated = cv.dilate(edges, kernel, iterations=1)

# cv.imshow('Dilated Edges', edges_dilated)

cv.waitKey()
