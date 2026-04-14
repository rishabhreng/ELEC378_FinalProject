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
from torchvision import transforms
import lightning as L

'''
Common functions to be used in both the HOG-SVM and CNN classifiers as well as model specific utility functions.
'''


FILE_PATH = Path(__file__).resolve().parent # To get the directory of our Python code
TRAIN_CSV_PATH = FILE_PATH / "train.csv" # Path to the train.csv file
TRAIN_IMAGE_PATH = FILE_PATH / "train_images" / "train_images" # Path to the train_images directory
TEST_IMAGE_PATH = FILE_PATH / "test_images" / "test_images" # Path to the test_images directory
RUNS_DIR = FILE_PATH / "runs" # Our trained models and logs

SEED = 42 # For reproducibility - answer to the ultimate question of life, the universe, and everything
IMAGE_SIZE = 224 # Resize images to square for our classifiers just like we did for YOLO


CLASSICAL_FEATURE_SIZE = 160 # Reduced dimensionality feature size, ADJUST AS NEEDED FOR PERFORMANCE/ACCURACY TRADEOFF


@dataclass(frozen=True)
class DatasetInfo:
    train_df: pd.DataFrame # pandas df containing training data
    val_df: pd.DataFrame # pandas df containing validation data
    class_names: list[str] # names of classes in the dataset
    class_to_index: dict[str, int] # map class names to integers for standardization

# Function to set seeds for all models for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # disable stochasticity
    torch.backends.cudnn.benchmark = False # disable autotune

# Function to load the metadata from given CSV files to read it as a pandas df
def load_metadata() -> DataFrame:
    df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(df)} rows of {df['TARGET'].nunique()} classes")
    return df

# Function to split the df into training and validation sets as given in the files
def split_dataset(df: DataFrame, val_size: float, random_state: int) -> DatasetInfo:
    # sklearn function to split dataset into training and validation
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["TARGET"], # standardization
        random_state=random_state,
    )
    class_names = sorted(df["TARGET"].unique()) # Sort class names for standardization
    class_to_index = {name: index for index, name in enumerate(class_names)} # Map class names to integers for standardization
    print(f"Train images: {len(train_df)}, Validation images: {len(val_df)}")
    # Return the split dataframes and useful metadata as a DatasetMetadata we defined earlier
    return DatasetInfo(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        class_names=class_names,
        class_to_index=class_to_index,
    )

# Function to load an image and resize it as square with RGB encoding
def load_image(filename: str, size: int = IMAGE_SIZE) -> Image.Image:
    return Image.open(TRAIN_IMAGE_PATH / filename).convert("RGB").resize((size, size))

# Function to open an image for the test set (doing the same resizing and RGB encoding as the training images)
def load_test_image(image_id: str, size: int = IMAGE_SIZE) -> Image.Image:
    return Image.open(TEST_IMAGE_PATH / f"{image_id}.jpg").convert("RGB").resize((size, size))

# Function to load the submission IDs from test images directory
def get_submission_image_ids() -> list[str]:
    return [path.stem for path in sorted(list(Path('./test_images/test_images/').glob('*.jpg')))]

# Function to compute the class weights for our CNN loss function
def compute_class_weights(train_df: pd.DataFrame, class_names: list[str]) -> torch.Tensor:
    counts = train_df["TARGET"].value_counts().reindex(class_names).to_numpy(dtype=np.float32) # how many images of each class are in the training set?
    weights = counts.sum() / (len(class_names) * counts)
    return torch.tensor(weights, dtype=torch.float32) # increase weight for classes with fewer images! exploration over exploitation!


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_to_index: dict[str, int], # mapping from class names to integers for standardization
        transform: transforms.Compose | None, # which transforms to apply to our images (optional)
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index] # row of the df for the index (reverse mapping from index to image and label)
        image = load_image(row["file_name"]) # load the image
        # apply transform if we have any (like normalization)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.class_to_index[row["TARGET"]], dtype=torch.int64) # convert the label to a tensor (integer encoding of class name)
        return image, label


class TestDataset(Dataset):
    def __init__(self, image_ids: Iterable[str], transform: transforms.Compose | None) -> None:
        self.image_ids = list(image_ids) # list of ids for the test set
        self.transform = transform # which transform to apply to our images (optional)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_id = self.image_ids[index] # get the image id for the index
        image = load_test_image(image_id) # load the test image
        # apply transform if we have any (like normalization)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id


class ConvNeurNetwork(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__() # initialize the Module
        # features of our convolutional layers (inspired from approaches found on google, adjust if necessary for performance/accuracy tradeoff)
        self.features = torch.nn.Sequential(
            # first layer - from rgb to 64 channels
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            # second layer - from 64 channels to 64 channels
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.10),
            # third layer - from 64 channels to 128 channels
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            # fourth layer - from 128 channels to 128 channels
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.15),
            # fifth layer - from 128 channels to 256 channels
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            # sixth layer - from 256 channels to 256 channels
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.20),
            # seventh layer - from 256 channels to 384 channels
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True),
            # adaptive average pooling to get fixed size output
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        # map the 384 features from the layers to the number of classes we have
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.4), # for regularization to prevent overfitting, adjust as necessary for performance/accuracy tradeoff
            torch.nn.Linear(384, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes),
        )

    # forward pass - the input goes through the layers to extract the features
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))

# Custom transform to remove background using GrabCut algorithm
class RemoveBackground:
    """Remove background from butterfly/moth images using GrabCut algorithm."""
    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            arr = np.array(img).astype(np.uint8)
            
            h, w = arr.shape[:-1]
            rect = (int(0.01 * w), int(0.01 * h), int(0.99 * w), int(0.99 * h))
            
            mask = np.zeros((h, w), np.uint8)
            cv2.grabCut(arr, mask, rect, np.zeros((1, 65)), np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3)))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
            
            arr[fg_mask == 0] = 255
            return Image.fromarray(arr)
        except Exception as e:
            # If background removal fails, return original image
            print(f"Warning: Background removal failed: {e}. Using original image.")
            return img

# Function to build the training and validation pipelines for our CNN (tensor conversion and normalization)
def build_neural_transforms(train: bool) -> transforms.Compose:
    # Use 0.5 mean/std for simpler normalization on butterfly data
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if train:
        return transforms.Compose(
            # for training, apply moderate transforms to prevent overfitting
            [
                RemoveBackground(),  # Remove background first
                transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),  # Slight upscale
                transforms.RandomCrop(IMAGE_SIZE),  # Random crop instead of resized crop
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),  # Reduced from 18 to 10 degrees,
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandAugment(2, 10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),  # Reduced intensity
                transforms.ToTensor(),
                normalize,
            ]
        )
    # else just convert to tensor and normalize
    return transforms.Compose([
        RemoveBackground(),  # Remove background first
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

# used for HOG-SVM classifier to build the feature matrix for training and validation
class FeatureExtractor:
    def __init__(self, feature_size: int = CLASSICAL_FEATURE_SIZE) -> None:
        self.feature_size = feature_size
        # HOG parameters inspired from common approaches found on google, adjust as necessary for performance/accuracy tradeoff
        self.hog = cv2.HOGDescriptor(
            _winSize=(feature_size, feature_size),
            _blockSize=(32, 32),
            _blockStride=(16, 16),
            _cellSize=(16, 16),
            _nbins=9,
        )
    # Function to convert an image to a feature vector using HOG
    def transform(self, image: Image.Image) -> np.ndarray:
        resized = np.array(image.resize((self.feature_size, self.feature_size)), dtype=np.uint8)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) # convert to grayscale for HOG computation
        hog_features = self.hog.compute(gray).reshape(-1)
        histograms: list[np.ndarray] = []
        # compute color histogram features for each channel - used for the HOG approach
        for channel in cv2.split(resized):
            channel_hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).reshape(-1)
            histograms.append(channel_hist.astype(np.float32)) 
        # do not forget to reappend the colors we lost when we converted to grayscale for the feature extraction
        color_features = np.concatenate(histograms)
        return np.concatenate([hog_features.astype(np.float32), color_features])

# Function to build the feature matrix for HOG classifier
def build_classical_feature_matrix(df: pd.DataFrame, extractor: FeatureExtractor) -> np.ndarray:
    features = [extractor.transform(Image.open(TRAIN_IMAGE_PATH / row.file_name).convert("RGB")) for row in df.itertuples(index=False)]
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
        
    def setup(self, stage: str = None) -> None:
        """Load and prepare data for training, validation, and testing."""
        df = load_metadata()
        dataset_info = split_dataset(df, val_size=self.val_size, random_state=self.seed)
        # Load metadata and split into train/val
        self.class_names = dataset_info.class_names
        # Compute class weights for loss function
        self.train_df = dataset_info.train_df
        self.val_df = dataset_info.val_df
        self.class_weights = compute_class_weights(self.train_df, self.class_names)

        self.class_to_index = dataset_info.class_to_index
        
        # Create train and val datasets
        self.train_dataset = ImageDataset(
            self.train_df,
            self.class_to_index,
            build_neural_transforms(train=True)
        )
        
        self.val_dataset = ImageDataset(
            self.val_df,
            self.class_to_index,
            build_neural_transforms(train=False)
        )
        
        # Load test dataset for prediction
        image_ids = get_submission_image_ids()
        self.test_dataset = TestDataset(image_ids, transform=build_neural_transforms(train=False))
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )