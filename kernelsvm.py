import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog
from tqdm import tqdm

# Configuration
DATA_DIR = os.getcwd()
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
RANDOM_STATE = 42
IMG_SIZE = 64
VAL_SPLIT = 0.2

def load_metadata(csv_path=CSV_PATH):
    """Load training metadata from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loading {len(df)} entries, {df['TARGET'].nunique()} classes")
    return df

def extract_features(filename, flip=False, img_dir=TRAIN_IMG_DIR):
    """Extract HOG and color histogram features from an image."""
    path = os.path.join(img_dir, filename)
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)

    if flip:
        img = np.fliplr(img)

    hog_features = hog(img, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), channel_axis=-1)

    hist_r, _ = np.histogram(img[:, :, 0], bins=32, range=(0, 256))
    hist_g, _ = np.histogram(img[:, :, 1], bins=32, range=(0, 256))
    hist_b, _ = np.histogram(img[:, :, 2], bins=32, range=(0, 256))

    color_features = np.concatenate([hist_r, hist_g, hist_b])
    color_features = color_features / (color_features.sum() + 1e-7)

    return np.concatenate([hog_features, color_features])

def predict_test_images():
    """Load test images and generate predictions."""
    test_img_dir = os.path.join(DATA_DIR, "test_images")
    test_files = sorted(os.listdir(test_img_dir))

    predictions = []
    for filename in tqdm(test_files, total=len(test_files)):
        feat = extract_features(filename, flip=False, img_dir=test_img_dir)
        if feat is not None:
            feat_scaled = scaler.transform([feat])
            feat_pca = pca.transform(feat_scaled)
            pred_enc = svc.predict(feat_pca)[0]
            pred_label = enc.inverse_transform([pred_enc])[0]
            predictions.append((filename.removesuffix('.jpg'), pred_label))

    # Save predictions to CSV
    pred_df = pd.DataFrame(predictions, columns=["ID", "TARGET"])
    print(pred_df.head())
    pred_df.to_csv("submission.csv", index=False)
    print("Predictions saved to submission.csv")


if __name__ == "__main__":
    metadata = load_metadata()

    enc = LabelEncoder()
    metadata['TARGET_ENC'] = enc.fit_transform(metadata['TARGET'])

    train_df, test_df = train_test_split(
        metadata, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=metadata['TARGET_ENC']
    )
    print(f"Train set: {len(train_df)} images, Validation set: {len(test_df)} images")

    # Extract training features (with augmentation)
    X_train, y_train = [], []
    print("Extracting train features...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        feat = extract_features(row["file_name"], flip=False, img_dir=TRAIN_IMG_DIR)
        if feat is not None:
            X_train.append(feat)
            y_train.append(row["TARGET_ENC"])

        feat_flipped = extract_features(row["file_name"], flip=True, img_dir=TRAIN_IMG_DIR)
        if feat_flipped is not None:
            X_train.append(feat_flipped)
            y_train.append(row["TARGET_ENC"])

    # Extract validation features
    X_val, y_val = [], []
    print("Extracting validation features...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        feat = extract_features(row["file_name"], flip=False, img_dir=TRAIN_IMG_DIR)
        if feat is not None:
            X_val.append(feat)
            y_val.append(row["TARGET_ENC"])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    print(f"Train samples (with augmentation): {len(X_train)}, Validation samples: {len(X_val)}")

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print("Features normalized")

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    print(f"PCA: reduced from {X_train.shape[1]} to {X_train_pca.shape[1]} components")

    # Train RBF SVM
    print("Training RBF SVM...")
    start_time = time.time()
    svc = SVC(kernel='rbf', cache_size=5000, random_state=RANDOM_STATE)
    svc.fit(X_train_pca, y_train)
    train_time = time.time() - start_time
    print(f"Trained in {train_time:.2f} seconds")

    # Evaluate on training and validation sets
    train_acc = svc.score(X_train_pca, y_train)
    val_acc = svc.score(X_val_pca, y_val)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    predict_test_images()
