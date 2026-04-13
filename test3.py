import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
img = Image.open('./train_images/train_images/train_000201.jpg')

# Convert PIL image to OpenCV format
img_cv = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
plt.imshow(img_cv)
# Convert to HSV for better color-based segmentation
hsv = cv.cvtColor(img_cv, cv.COLOR_BGR2HSV)

# Define range for blue colors (butterfly wings)
# Blue in HSV: H ~100-130, S ~100-255, V ~100-255
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create mask for blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)

# Morphological operations to clean up the mask
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

# Apply morphological closing to connect nearby regions (butterfly body)
kernel_larger = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_larger, iterations=1)

# Apply mask to original image
img_no_bg = img_cv.copy()
img_no_bg[mask == 0] = [255, 255, 255]  # White background

# Extract dominant colors from butterfly only (non-white pixels)
butterfly_pixels = []
for i in range(img_no_bg.shape[0]):
    for j in range(img_no_bg.shape[1]):
        if not np.all(img_no_bg[i, j] == [255, 255, 255]):
            # Convert BGR to RGB
            butterfly_pixels.append(img_no_bg[i, j][::-1])

butterfly_pixels = np.array(butterfly_pixels)
print(f"Number of butterfly pixels: {len(butterfly_pixels)}")
if len(butterfly_pixels) > 0:
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(butterfly_pixels)
    
    # Get dominant colors sorted by frequency
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors_butterfly = kmeans.cluster_centers_[sorted_indices].astype(int)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Butterfly without background
    img_display = cv.cvtColor(img_no_bg, cv.COLOR_BGR2RGB)
    axes[1].imshow(img_display)
    axes[1].set_title('Butterfly (Background Removed)')
    axes[1].axis('off')
    
    # Color palette
    axes[2].set_title('Color Palette (Butterfly Only - Top 8)')
    axes[2].set_xlim(0, 8)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    
    for i, color in enumerate(dominant_colors_butterfly):
        normalized_color = color / 255.0
        rect = mpatches.Rectangle((i, 0), 1, 1, facecolor=normalized_color)
        axes[2].add_patch(rect)
        axes[2].text(i + 0.5, -0.1, f'RGB{tuple(color)}', ha='center', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nDominant colors in BUTTERFLY ONLY (RGB):")
    for i, color in enumerate(dominant_colors_butterfly, 1):
        print(f"  {i}. {tuple(color)}")