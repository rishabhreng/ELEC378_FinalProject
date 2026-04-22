import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description='Train kNN on flower images and predict one test picture.')
    parser.add_argument('--train-csv', default='train.csv', help='Path to train.csv')
    parser.add_argument('--train-dir', default='train_images/train_images', help='Path to training images directory')
    parser.add_argument('--test-dir', default='test_images/test_images', help='Path to test images directory')
    parser.add_argument('--test-file', default='test_000057.jpg', help='Test image filename or ID to classify')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for kNN')
    parser.add_argument('--resize', type=int, default=32, help='Image side length to resize to')
    parser.add_argument('--max-train', type=int, default=5000, help='Maximum number of training images to use')
    return parser.parse_args()


def resolve_test_filename(test_file: str) -> str:
    if test_file.lower().startswith('test_picture'):
        digits = re.search(r'(\d+)', test_file)
        if digits:
            return f"test_{int(digits.group(1)):06d}.jpg"
    if not test_file.lower().endswith('.jpg'):
        return f"{test_file}.jpg"
    return test_file


def load_image(path: str, size: int) -> np.ndarray:
    with Image.open(path) as img:
        image = img.convert('L').resize((size, size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32).ravel() / 255.0


def load_training_data(train_csv: str, train_dir: str, resize: int, max_train: int):
    df = pd.read_csv(train_csv)
    if 0 < max_train < len(df):
        df = df.sample(n=max_train, random_state=42).reset_index(drop=True)

    X = []
    y = []

    for _, row in df.iterrows():
        filename = row['file_name']
        label = row['TARGET']
        image_path = os.path.join(train_dir, filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Training image not found: {image_path}')

        X.append(load_image(image_path, resize))
        y.append(label)

    X = np.stack(X)
    return X, np.array(y, dtype=object)


def main():
    args = parse_args()
    args.test_file = resolve_test_filename(args.test_file)
    test_path = os.path.join(args.test_dir, args.test_file)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f'Test image not found: {test_path}')

    print('Loading training data...')
    X_train, y_train = load_training_data(args.train_csv, args.train_dir, args.resize, args.max_train)
    print(f'Loaded {len(X_train)} training images.')

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    print('Scaling features...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f'Training kNN (k={args.k})...')
    model = KNeighborsClassifier(
        n_neighbors=args.k,
        weights='distance',
        algorithm='ball_tree',
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train_encoded)

    print(f'Predicting {args.test_file}...')
    X_test = load_image(test_path, args.resize).reshape(1, -1)
    X_test_scaled = scaler.transform(X_test)
    prediction_encoded = model.predict(X_test_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    probabilities = model.predict_proba(X_test_scaled)[0]
    top_indices = np.argsort(probabilities)[::-1][:5]
    top_labels = label_encoder.inverse_transform(top_indices)
    top_probs = probabilities[top_indices]

    print(f'Prediction for {args.test_file}: {prediction_label}')
    print('Top 5 class probabilities:')
    for label, prob in zip(top_labels, top_probs):
        print(f'  {label}: {prob:.4f}')


if __name__ == '__main__':
    main()
