"""Train YOLO-11 on the butterfly dataset and generate submission predictions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


FILE_PATH = Path(__file__).resolve().parent
TRAIN_CSV_PATH = FILE_PATH / "train.csv"
SAMPLE_SUBMISSION_PATH = FILE_PATH / "sample_submission.csv"
TRAIN_IMAGE_PATH = FILE_PATH / "train_images" / "train_images"
TEST_IMAGE_PATH = FILE_PATH / "test_images" / "test_images"
PREPARED_DATA_DIR = FILE_PATH / "yolo_classification_data"
RUNS_DIR = FILE_PATH / "runs"

RANDOM_STATE = 42 # seed for reproducibility
DEFAULT_MODEL = "yolo11n-cls.pt" # nano model


# Path and class information for the prepared dataset, returned by prepare_yolo_dataset()
@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    train_dir: Path
    val_dir: Path
    class_names: list[str]


# Method to load the training metadata from train.csv
def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(df)} rows across {df['TARGET'].nunique()} classes")
    return df

# Method to load an image from filename
def load_image(filename: str, size: int = 224) -> Image.Image:
    image_path = TRAIN_IMAGE_PATH / filename
    return Image.open(image_path).convert("RGB").resize((size, size)) # resize to match yolo model input size


# Method to display a sample image from the dataset with its label
def show_sample_image(df: pd.DataFrame, index: int, size: int = 224) -> None:
    row = df.iloc[index] # get the row at the specified index
    image = load_image(row["file_name"], size=size)
    plt.imshow(image) # display the image
    plt.title(f"[{index}] {row['TARGET']}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Method to create a shortcut to the training images for YOLO model to use
def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
        dst.symlink_to(src.resolve())


# Method to prepare dataset for YOLO classification
def prepare_yolo_dataset(
    df: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> PreparedDataset:
    train_df, val_df = train_test_split(
        df, # split the dataframe into train and validation sets
        test_size=val_size,
        stratify=df["TARGET"],
        random_state=random_state,
    )
    shutil.rmtree(PREPARED_DATA_DIR) # clear out any existing prepared data
    train_dir = PREPARED_DATA_DIR / "train"
    val_dir = PREPARED_DATA_DIR / "val"
    class_names = sorted(df["TARGET"].unique()) # get the unique class names from the TARGET column

    # Create directories for YOLO format
    for class_name in class_names:
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    # Copy the images to those directories (their shortcuts to save space actually)
    for split_df, split_dir in ((train_df, train_dir), (val_df, val_dir)):
        for row in split_df.itertuples(index=False):
            source_path = TRAIN_IMAGE_PATH / row.file_name
            destination_path = split_dir / row.TARGET / row.file_name
            link_or_copy(source_path, destination_path)

    print(f"Prepared YOLO dataset at {PREPARED_DATA_DIR}")
    print(f"Train images: {len(train_df)}, Validation images: {len(val_df)}")
    return PreparedDataset(
        root=PREPARED_DATA_DIR,
        train_dir=train_dir,
        val_dir=val_dir,
        class_names=class_names,
    )

# Method to train a YOLO classifier on the prepared dataset
def train_yolo_classifier(
    dataset: PreparedDataset, # dataset info returned by prepare_yolo_dataset()
    model_name: str, # use a non-pretrained yolo model
    epochs: int, # number of training epochs
    imgsz: int, # image size
    batch: int, # batch size
    device: str | None, # use GPU if available, otherwise CPU
    workers: int, # parallelize
    project_dir: Path, # directory where training runs are stored
    run_name: str, # name of run for saving results
    pretrained: bool, # set to False to train from scratch
) -> Path:
    model = YOLO(model_name) # load the yolo model
    train_kwargs: dict[str, object] = {
        "data": str(dataset.root),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "project": str(project_dir),
        "name": run_name,
        "patience": 4, # to prevent overfitting, return after 4 epochs of convergence
        "pretrained": pretrained, # toggle pretrained initialization here
        "verbose": True, # print progress to console to not get mad
    }
    if device is not None:
        train_kwargs["device"] = device # use gpu if specified!

    results = model.train(**train_kwargs) # train the model
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    print(f"Best weights saved to {best_weights}")
    return best_weights

# Method to run YOLO prediction/classification
def predict_label(result) -> str:
    class_index = int(result.probs.top1)
    names = result.names
    return str(names[class_index]) # get the predicted class name from the result

# Generate the csv file for classified test images
def generate_submission(weights_path: Path, output_path: Path, batch: int, imgsz: int) -> pd.DataFrame:
    model = YOLO(str(weights_path))
    submission_template = pd.read_csv(SAMPLE_SUBMISSION_PATH) # read the sample submission to get the IDs for the test images
    image_paths = [TEST_IMAGE_PATH / f"{sample_id}.jpg" for sample_id in submission_template["ID"]] # the same ids are gonna be used

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        batch=batch,
        verbose=False,
    )
    predictions = [predict_label(result) for result in results]

    submission = pd.DataFrame({"ID": submission_template["ID"], "TARGET": predictions}) # create a new dataframe with the IDs and the predicted labels
    submission.to_csv(output_path, index=False) # write and save the submission file to the specified path
    print(f"Wrote submission to {output_path}")
    return submission


def main() -> None:
    SHOW_SAMPLE = False # display sample image from dataset (debugging)
    TRAIN_MODEL = True # train a model or use an existing one
    USE_PRETRAINED = False # never switch to true for the sake of the assignment
    MODEL_NAME = "yolo11n-cls.yaml" # classifier -cls version of yolo nano
    EPOCHS = 2 # increase this...
    IMGSZ = 224 # most yolo models need square images
    BATCH = 32 # increase batch size for faster convergence if you have a gpu
    VAL_SIZE = 0.2 # 20% is a good rule of thumb for validation ratio
    SEED = RANDOM_STATE # 42 is the answer to the ultimate question of life, the universe, and everything.
    DEVICE = None # set to "cuda" to use gpu
    WORKERS = 8 # assume 8 smt cores
    PROJECT_DIR = RUNS_DIR # where the training runs are stored
    RUN_NAME = "yolo11_butterflies"
    OUTPUT_PATH = FILE_PATH / "submission.csv"
    SAMPLE_INDEX = 0

    print("Loading metadata")
    df = load_metadata()
    if SHOW_SAMPLE:
        show_sample_image(df, index=SAMPLE_INDEX)

    dataset = prepare_yolo_dataset(df, val_size=VAL_SIZE, random_state=SEED)
    if TRAIN_MODEL:
        weights_path = train_yolo_classifier(
            dataset=dataset,
            model_name=MODEL_NAME,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project_dir=PROJECT_DIR,
            run_name=RUN_NAME,
            pretrained=USE_PRETRAINED,
        )

    generate_submission(
        weights_path=weights_path,
        output_path=OUTPUT_PATH,
        batch=BATCH,
        imgsz=IMGSZ,
    )


if __name__ == "__main__":
    main()
