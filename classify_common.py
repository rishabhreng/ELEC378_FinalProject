# import a bunch of stuff
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
import lightning as L

"""
Common functions to be used in both the HOG-SVM and CNN classifiers as well as model specific utility functions.
"""


FILE_PATH = Path(__file__).resolve().parent
TRAIN_CSV_PATH = FILE_PATH / "train.csv"
TRAIN_IMAGE_PATH = FILE_PATH / "train_images_cropped"
TEST_IMAGE_PATH = FILE_PATH / "test_images_cropped"
RUNS_DIR = FILE_PATH / "runs"

SEED = 42  # For reproducibility - answer to the ultimate question of life, the universe, and everything
IMAGE_SIZE = 224
CLASSICAL_FEATURE_SIZE = 160  # Reduced dimensionality feature size, ADJUST AS NEEDED FOR PERFORMANCE/ACCURACY TRADEOFF


@dataclass(frozen=True)
class DatasetInfo:
    train_df: pd.DataFrame  # pandas df containing training data
    val_df: pd.DataFrame  # pandas df containing validation data
    class_names: list[str]  # names of classes in the dataset
    class_to_index: dict[str, int]  # map class names to integers for standardization


# Function to load the metadata from train.csv
def load_metadata() -> DataFrame:
    df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(df)} rows of {df['TARGET'].nunique()} classes")
    return df


# Split dataset into train/val with stratification
def split_dataset(df: DataFrame, val_size: float, random_state: int) -> DatasetInfo:
    # sklearn function to split dataset into training and validation
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["TARGET"],
        random_state=random_state,
    )
    class_names = sorted(df["TARGET"].unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    print(f"Train images: {len(train_df)}, Validation images: {len(val_df)}")
    return DatasetInfo(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        class_names=class_names,
        class_to_index=class_to_index,
    )


# Load and resize image to RGB
def load_image(filename: str, size: int = IMAGE_SIZE) -> Image.Image:
    return cv2.cvtColor(cv2.imread(str(TRAIN_IMAGE_PATH / filename)), cv2.COLOR_BGR2RGB)


# Load test image with same preprocessing as training
def load_test_image(image_id: str, size: int = IMAGE_SIZE) -> Image.Image:
    return cv2.cvtColor(
        cv2.imread(str(TEST_IMAGE_PATH / f"{image_id}.jpg")), cv2.COLOR_BGR2RGB
    )


# Load submission image IDs from test directory
def get_submission_image_ids() -> list[str]:
    return [path.stem for path in sorted(list(Path(TEST_IMAGE_PATH).glob("*.jpg")))]


# Compute class weights to handle class imbalance
def compute_class_weights(
    train_df: pd.DataFrame, class_names: list[str]
) -> torch.Tensor:
    counts = (
        train_df["TARGET"]
        .value_counts()
        .reindex(class_names)
        .to_numpy(dtype=np.float32)
    )
    weights = counts.sum() / (len(class_names) * counts)
    return torch.tensor(weights, dtype=torch.float32)


class ImageDataset(Dataset):
    """Dataset for loading training and validation images with labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_index: dict[str, int],
        transform: transforms.Compose | None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image = load_image(row["file_name"])
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.class_to_index[row["TARGET"]], dtype=torch.int64)
        return image, label


class TestDataset(Dataset):
    """Dataset for loading test images without labels."""

    def __init__(
        self, image_ids: Iterable[str], transform: transforms.Compose | None
    ) -> None:
        self.image_ids = list(image_ids)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_id = self.image_ids[index]
        image = load_test_image(image_id)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # The Skip Connection
        return torch.nn.functional.relu(out)


class CNN(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            torch.nn.Dropout2d(0.25),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Dropout(0.5), torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Build training/validation data augmentation and preprocessing pipeline
def build_neural_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(
                    (int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1)), antialias=True
                ),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandAugment(2, 10),
                transforms.ToDtype(torch.float32, scale=True),  # Scales to [0,1]
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                ),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


# used for HOG-SVM classifier to build the feature matrix for training and validation
class FeatureExtractor:
    """Extract HOG and color histogram features from images for classical ML."""

    def __init__(self, feature_size: int = CLASSICAL_FEATURE_SIZE) -> None:
        self.feature_size = feature_size
        self.hog = cv2.HOGDescriptor(
            _winSize=(feature_size, feature_size),
            _blockSize=(32, 32),
            _blockStride=(16, 16),
            _cellSize=(16, 16),
            _nbins=9,
        )

    def transform(self, image: Image.Image) -> np.ndarray:
        resized = np.array(
            image.resize((self.feature_size, self.feature_size)), dtype=np.uint8
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        hog_features = self.hog.compute(gray).reshape(-1)
        histograms = []
        for channel in cv2.split(resized):
            channel_hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).reshape(
                -1
            )
            histograms.append(channel_hist.astype(np.float32))
        color_features = np.concatenate(histograms)
        return np.concatenate([hog_features.astype(np.float32), color_features])


# Build feature matrix for HOG-based classification
def build_classical_feature_matrix(
    df: pd.DataFrame, extractor: FeatureExtractor
) -> np.ndarray:
    features = [
        extractor.transform(Image.open(TRAIN_IMAGE_PATH / row.file_name).convert("RGB"))
        for row in df.itertuples(index=False)
    ]
    return np.stack(features, axis=0)


class ButterflyDataModule(L.LightningDataModule):
    """LightningDataModule for loading butterfly/moth image classification data."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        val_size: float = 0.2,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.seed = seed
        self._setup_stage_done: set[str] = set()

    def setup(self, stage: str = None) -> None:
        """Load and prepare data for training, validation, and testing."""
        if stage in (None, "fit", "validate") and "fit" not in self._setup_stage_done:
            df = load_metadata()
            dataset_info = split_dataset(
                df, val_size=self.val_size, random_state=self.seed
            )
            # Load metadata and split into train/val
            self.class_names = dataset_info.class_names
            # Compute class weights for loss function
            self.train_df = dataset_info.train_df
            self.val_df = dataset_info.val_df
            self.class_weights = compute_class_weights(self.train_df, self.class_names)

            self.class_to_index = dataset_info.class_to_index

            # Create train and val datasets
            self.train_dataset = ImageDataset(
                self.train_df, self.class_to_index, build_neural_transforms(train=True)
            )

            self.val_dataset = ImageDataset(
                self.val_df, self.class_to_index, build_neural_transforms(train=False)
            )
            self._setup_stage_done.add("fit")

        if (
            stage in (None, "predict", "test")
            and "predict" not in self._setup_stage_done
        ):
            # Load test dataset for prediction
            image_ids = get_submission_image_ids()
            self.test_dataset = TestDataset(
                image_ids, transform=build_neural_transforms(train=False)
            )
            self._setup_stage_done.add("predict")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
