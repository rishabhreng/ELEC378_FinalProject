from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

# parser = argparse.ArgumentParser(description='Code for Image Segmentation with Distance Transform and Watershed Algorithm.\
#     Sample code showing how to segment overlapping objects using Laplacian filtering, \
#     in addition to Watershed and Distance Transformation')
# parser.add_argument('--input', help='Path to input image.', default='cards.png')
# args = parser.parse_args()

src = cv.imread(cv.samples.findFile('./test_images/test_images/test_000999.jpg'))
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)

# Show source image
cv.imshow('Source Image', src)

src[np.all(src == 255, axis=2)] = 0

# Show output image
# cv.imshow('Black Background Image', )

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian

# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)

#cv.imshow('Laplace Filtered Image', imgLaplacian)
cv.imshow('New Sharped Image', imgResult)

# Use GrabCut for foreground segmentation
mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Define rectangle for initial region (approximate foreground area)
height, width = src.shape[:2]
rect = (10, 10, width-10, height-10)

# Apply GrabCut
cv.grabCut(src, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# Create binary mask from GrabCut output
# cv.GC_PR_FGD (probably foreground) = 1, cv.GC_FGD (foreground) = 3
mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
cv.imshow('GrabCut Mask', mask2)

# Apply mask to original image
dst = cv.bitwise_and(src, src, mask=mask2)

# Visualize the final image
cv.imshow('Final Result', dst)

cv.waitKey()