import os
import matplotlib.pyplot as plt

# Path to the train_images directory
train_images_dir = 'train_images/train_images/'

# List to store file sizes
file_sizes = []

# Get list of files in the directory
for filename in os.listdir(train_images_dir):
    # Check if it's an image file (you can add more extensions if needed)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        filepath = os.path.join(train_images_dir, filename)
        size = os.path.getsize(filepath)
        file_sizes.append((filename, size))

# Optional: print summary
if file_sizes:
    total_size = sum(size for _, size in file_sizes)
    avg_size = total_size / len(file_sizes)
    print(f"Total images: {len(file_sizes)}")
    print(f"Total size: {total_size} bytes")
    print(f"Average size: {avg_size:.2f} bytes")
    # Print first 10 file sizes as example
    print("\nFirst 10 file sizes:")
    for filename, size in file_sizes[:10]:
        print(f"{filename}: {size} bytes")
else:
    print("No image files found.")
sorted_file_sizes = sorted(file_sizes, key=lambda x: x[1])
plt.scatter(range(len(file_sizes)), [size for _, size in sorted_file_sizes])
plt.title("File Sizes of Training Images")
plt.xlabel("Image Index (sorted by size)")
plt.ylabel("File Size (bytes)")
plt.show()