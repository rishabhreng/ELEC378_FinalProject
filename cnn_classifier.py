# import a bunch of stuff
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# import a bunch of stuff from our common module
from classify_common import (
    FILE_PATH,
    IMAGE_SIZE,
    SEED,
    RUNS_DIR,
    ConvNeurNetwork,
    TestDataset,
    build_neural_loaders,
    build_neural_transforms,
    compute_class_weights,
    load_metadata,
    load_submission_ids,
    set_seed,
    split_dataset,
)

# Function to train the CNN for one epoch and return average loss and error rate
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train() # train using pytorch's training mode
    avg_loss = 0.0
    avg_correct = 0
    total = 0
    # use cuda if possible
    use_amp = device.type == "cuda"
    # cuda stuff for better performance
    for images, labels in loader:
        images = images.to(device, non_blocking=True) # move everything to either cuda or cpu
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True) # set the gradients to none rather than zero for even betterperformance
        # optimization step
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images) # get the logits from the model
            loss = criterion(logits, labels) # compute cross entropy loss
        loss.backward() # back propagate the loss
        optimizer.step()

        # update avg loss and accuracy for the epoch
        batch_size = labels.size(0)
        avg_loss += loss.item() * batch_size
        avg_correct += (logits.argmax(dim=1) == labels).sum().item() # count the number of correct predictions in the batch
        total += batch_size

    return avg_loss / total, avg_correct / total


@torch.no_grad() # no need to compute gradients for evaluation
# Function to score the CNN on the validation set and return avg loss and error rate
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    avg_loss = 0.0
    avg_correct = 0
    total = 0
    # logic same as above without backpropagation and optimization since we're evaluating
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        avg_loss += loss.item() * batch_size
        avg_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return avg_loss / total, avg_correct / total

# Function to train the CNN, main pipeline for training and saving best model checkpoint
def train_neural_classifier(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    num_workers: int,
    patience: int,
    run_name: str,
) -> Path:
    df = load_metadata() # load the metadata into a pandas df
    dataset = split_dataset(df, val_size=0.2, random_state=SEED) # split the df into train and validation sets

    # create directory for the run and define model path
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoints = run_dir / "best_cnn.pt"

    # use cuda if possible
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    load_train, load_validate = build_neural_loaders(dataset, batch_size=batch_size, num_workers=num_workers) # build data loaders for both sets
    # create the model, loss function, optimization regime and learning rate scheduler
    model = ConvNeurNetwork(num_classes=len(dataset.class_names)).to(torch_device)
    class_weights = compute_class_weights(dataset.train_df, dataset.class_names).to(torch_device) # compute the class weights so we explore while also exploit
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05) # use cross entropy loss function with out computed weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # our optimization is adam with weight decay, change if necessary
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # we use cosine annealing here, change if necessary to optimize

    # save best model checkpoint
    best_val_accuracy = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    old_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, load_train, criterion, optimizer, torch_device)
        validation_loss, validation_accuracy = evaluate_classifier(model, load_validate, criterion, torch_device)
        scheduler.step()
        print(
            f"Epoch {epoch:03d} out of {epochs:03d}"
            f"training loss {train_loss:.4f} with accuracy {train_accuracy:.4f}"
            f"validation loss {validation_loss:.4f} with accuracy {validation_accuracy:.4f}"
        )
        # update best checkpoint if accuracy improves
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            # save best model checkpoint to cpu to avoid pytorch crashing from gpu memory outage
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": dataset.class_names,
                    "image_size": IMAGE_SIZE,
                },
                model_checkpoints,
            )
            old_epochs = 0
        else:
            # the accuracy stagnates.
            old_epochs += 1
        # early return when convergence occurs
        if old_epochs >= patience:
            print(f"Old epoch threshold reached, returned after {epoch} epochs")
            break

    return model_checkpoints


@torch.no_grad() # no need to compute gradients for prediction
# Function to predict labels for our test set using the trained model and save the submission file
def predict_neural_labels(weights_path: Path, output_path: Path, batch_size: int, device: str) -> pd.DataFrame:
    checkpoint = torch.load(weights_path, map_location="cpu") # load model checkpoint to cpu to avoid cuda memory outage
    class_names = list(checkpoint["class_names"]) # get class names from the checkpoint
    # use cuda if possible
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # set our model to evaluation (classification) mode and load weights from checkpoints
    model = ConvNeurNetwork(num_classes=len(class_names)).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # build test data loader and predict the labels in batches to avoid memory issues
    image_ids = load_submission_ids()
    test_dataset = TestDataset(image_ids, transform=build_neural_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # predict the labels for the test set
    predictions: list[str] = []
    for images, _ in test_loader:
        images = images.to(torch_device) # move the batch of images to either cuda or cpu
        logits = model(images) # get logits from model 
        # map predicted indices back to class names and add the predicted class names to our predictions list
        predicted_indices = logits.argmax(dim=1).cpu().numpy().tolist()
        predictions.extend(class_names[index] for index in predicted_indices)

    # generate submission CSV file with the predicted labels
    submission = pd.DataFrame({"ID": image_ids, "TARGET": predictions})
    submission.to_csv(output_path, index=False)
    print(f"Wrote submission.csv to {output_path}")
    return submission


def main() -> None:
    set_seed(SEED) # set the random seed (42) for reproducibility
    # train the CNN and save the best model checkpoint
    best_checkpoint = train_neural_classifier(
        # ADJUST HYPERPARAMETERS TO OPTIMIZE PERFORMANCE AND ACCURACY
        epochs=120,
        batch_size=64,
        learning_rate=3e-4,
        device="cuda",
        num_workers=8,
        patience=14,
        run_name="scratch_cnn_butterflies",
    )
    # predict the labels using the trained model and write the submission file
    predict_neural_labels(
        # ADJUST HYPERPARAMETERS TO OPTIMIZE PERFORMANCE AND ACCURACY
        weights_path=best_checkpoint,
        output_path=FILE_PATH / "submission_cnn.csv",
        batch_size=64,
        device="cuda",
    )


if __name__ == "__main__":
    main()