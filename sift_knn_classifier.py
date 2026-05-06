import os
import warnings

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
SIFT_DIM = 128
FEATURE_DIM = SIFT_DIM * 2
sift = cv.SIFT_create()

TRAIN_CSV = "train.csv"
TRAIN_DIR = "train_images"
TEST_DIR = "test_images"
SAMPLE_RATE = 5
NEIGHBORS = 5
MAX_TEST_ID = 1000
OUTPUT_CSV = "results/submission_sift_knn.csv"
RANDOM_STATE = 42
VAL_SPLIT = 0.2


def extract_sift_features(image_path):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def get_feature_vector(descriptors):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    descriptors = descriptors.astype(np.float32)
    mean_desc = np.mean(descriptors, axis=0)
    std_desc = np.std(descriptors, axis=0)
    return np.concatenate([mean_desc, std_desc]).astype(np.float32)

def build_training_set(train_df, train_image_dir, sample_rate):
    train_features = []
    train_labels = []

    print("Extracting SIFT features from training images...")
    iterator = tqdm(train_df.iterrows(), total=len(train_df))
    for idx, row in iterator:
        if idx % sample_rate != 0:
            continue
        img_path = os.path.join(train_image_dir, row["file_name"])
        feature_vector = get_feature_vector(extract_sift_features(img_path))
        train_features.append(feature_vector)
        train_labels.append(row["TARGET"])

    X = np.array(train_features, dtype=np.float32)
    y = np.array(train_labels)
    print(f"Extracted features shape: {X.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    return X_train, y_train, X_val, y_val


def predict_test_set(knn, scaler, test_image_dir, max_test_id):
    rows = []
    print("\nPredicting on test images...")

    for test_id in tqdm(range(max_test_id + 1)):
        filename = f"test_{test_id:06d}.jpg"
        img_path = os.path.join(test_image_dir, filename)

        if not os.path.exists(img_path):
            continue

        feature_vector = get_feature_vector(extract_sift_features(img_path)).reshape(1, -1)
        feature_vector = scaler.transform(feature_vector)
        prediction = knn.predict(feature_vector)[0]
        rows.append({"file_name": filename, "TARGET": prediction})

    return pd.DataFrame(rows)

def main():
    train_df = pd.read_csv(TRAIN_CSV)
    if "file_name" not in train_df.columns or "TARGET" not in train_df.columns:
        raise ValueError("Train CSV must contain 'file_name' and 'TARGET' columns.")

    X_train, y_train, X_val, y_val = build_training_set(train_df, TRAIN_DIR, SAMPLE_RATE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("\nTraining KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=NEIGHBORS)
    knn.fit(X_train, y_train)

    # Evaluate on training and validation sets
    train_acc = knn.score(X_train, y_train)
    val_acc = knn.score(X_val, y_val)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Predict on test set
    pred_df = predict_test_set(knn, scaler, TEST_DIR, MAX_TEST_ID)

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pred_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(pred_df)} predictions to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
