import cv2 as cv
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load train.csv to map filenames to labels
train_df = pd.read_csv('train.csv')
label_to_class = dict(zip(train_df['file_name'], train_df['TARGET']))

# Initialize SIFT
sift = cv.SIFT_create()

def extract_sift_features(image_path):
    """Extract SIFT features from an image"""
    img = cv.imread(image_path)
    if img is None:
        return None
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return descriptors

def get_feature_vector(descriptors):
    """Convert SIFT descriptors to a feature vector using statistics"""
    if descriptors is None or len(descriptors) == 0:
        # Return zero vector if no descriptors found
        return np.zeros(136)  # 128*1 for mean + 8*1 for other stats
    
    descriptors = descriptors.astype(np.float32)
    
    # Use mean and std of descriptors
    mean_desc = np.mean(descriptors, axis=0)
    std_desc = np.std(descriptors, axis=0)
    
    # Combine mean and std
    feature_vector = np.concatenate([mean_desc, std_desc])
    
    return feature_vector

# Extract features from training data (sample every few images for speed)
print("Extracting SIFT features from training images...")
train_features = []
train_labels = []

train_image_dir = 'train_images/train_images/'
sample_rate = 5  # Use every 5th image to speed up

for idx, row in train_df.iterrows():
    if idx % sample_rate != 0:  # Sample every 5th image
        continue
        
    img_path = os.path.join(train_image_dir, row['file_name'])
    descriptors = extract_sift_features(img_path)
    feature_vector = get_feature_vector(descriptors)
    
    train_features.append(feature_vector)
    train_labels.append(row['TARGET'])
    
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(train_df)} training images")

train_features = np.array(train_features)
print(f"Training features shape: {train_features.shape}")

# Standardize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

# Train KNN classifier
print("\nTraining KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_features, train_labels)

# Test on specific test images
test_indices = list(range(1, 11)) + list(range(127, 138))  # 1-10 and 127-137
test_image_dir = 'test_images/test_images/'

print("\nTesting on selected images:")
print("-" * 80)
print(f"{'Test ID':<15} {'Actual':<30} {'Predicted':<30} {'Match':<10}")
print("-" * 80)

ground_truth = {
    'test_000003.jpg': 'ADONIS',
    'test_000015.jpg': 'AFRICAN GIANT SWALLOWTAIL'
}

correct = 0
total = 0

for test_id in test_indices:
    filename = f'test_{test_id:06d}.jpg'
    img_path = os.path.join(test_image_dir, filename)
    
    if not os.path.exists(img_path):
        print(f"Warning: {filename} not found")
        continue
    
    # Extract SIFT features
    descriptors = extract_sift_features(img_path)
    feature_vector = get_feature_vector(descriptors).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)
    
    # Predict
    prediction = knn.predict(feature_vector)[0]
    
    # Get actual label if available
    actual = ground_truth.get(filename, 'Unknown')
    is_match = (prediction == actual) if actual != 'Unknown' else '?'
    
    if actual != 'Unknown':
        total += 1
        if is_match:
            correct += 1
    
    print(f"{test_id:<15} {actual:<30} {prediction:<30} {str(is_match):<10}")

print("-" * 80)
if total > 0:
    print(f"\nAccuracy on known labels: {correct}/{total} ({100*correct/total:.1f}%)")
else:
    print("\nNo ground truth labels to compare")
