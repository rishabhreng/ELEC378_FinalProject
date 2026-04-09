# import a bunch of stuff
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

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



class CNNButterflyClassifier(L.LightningModule):
    """Lightning wrapper for the CNN butterfly classifier with training, validation, and prediction steps."""
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.01,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = ConvNeurNetwork(num_classes=num_classes)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        
        # Register class weights as buffer so they move with the model
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        
        # Create loss function with class weights and label smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Use ReduceLROnPlateau for better plateau handling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # maximize accuracy
            factor=0.5,  # reduce LR by 50% when plateau detected
            patience=5,  # wait 5 epochs before reducing
            min_lr=1e-7,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
            }
        }


def train_neural_classifier(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    num_workers: int,
    patience: int,
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    class_weights: torch.Tensor,
    ckpt_to_start_from: Path | None = None,
) -> Path:
    """Train the CNN using Lightning Trainer and return path to best checkpoint."""
    
    # Create directory for the run
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the Lightning module
    model = CNNButterflyClassifier(
        num_classes=len(class_names),
        learning_rate=learning_rate,
        weight_decay=1e-4,
        label_smoothing=0.01,
        class_weights=class_weights,
    )
    
    # Load checkpoint if provided
    if ckpt_to_start_from is not None:
        checkpoint = torch.load(ckpt_to_start_from, map_location="cpu")
        model.cnn.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized model with weights from {ckpt_to_start_from}")
    
    # Define callbacks for early stopping and checkpointing
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        mode="max",
        verbose=True,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_cnn",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    
    # Create the Lightning Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=device if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[early_stop, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Return the best checkpoint path
    best_ckpt = checkpoint_callback.best_model_path
    
    # Convert Lightning checkpoint to the old format for compatibility with predict function
    lightning_ckpt = torch.load(best_ckpt, map_location="cpu")
    
    # Extract the model weights and metadata
    model_state_dict = {}
    for key in lightning_ckpt["state_dict"]:
        if key.startswith("cnn."):
            # Remove "cnn." prefix to get original model keys
            model_state_dict[key[4:]] = lightning_ckpt["state_dict"][key]
    
    # Save in the original format
    best_model_path = run_dir / "best_cnn.pt"
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
        },
        best_model_path,
    )
    
    print(f"Best model saved to {best_model_path}")
    return best_model_path


@torch.no_grad()
def predict_neural_labels(
    weights_path: Path,
    output_path: Path,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Predict labels for test set using trained model and save submission file."""
    
    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = list(checkpoint["class_names"])
    
    # Use cuda if possible
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Set model to evaluation mode and load weights
    model = ConvNeurNetwork(num_classes=len(class_names)).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Build test data loader
    image_ids = load_submission_ids()
    test_dataset = TestDataset(image_ids, transform=build_neural_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Predict labels for test set
    predictions: list[str] = []
    for images, _ in test_loader:
        images = images.to(torch_device)
        logits = model(images)
        predicted_indices = logits.argmax(dim=1).cpu().numpy().tolist()
        predictions.extend(class_names[index] for index in predicted_indices)
    
    # Generate submission CSV file
    submission = pd.DataFrame({"ID": image_ids, "TARGET": predictions})
    submission.to_csv(output_path, index=False)
    print(f"Wrote submission file to {output_path}")
    return submission


def main() -> None:
    torch.set_float32_matmul_precision('medium' if torch.cuda.is_available() else 'high')
    run_name = "scratch_cnn_butterflies5"
    device = "cuda"
    batch_size = 32
    num_workers = 4
    
    # Set the random seed for reproducibility
    set_seed(SEED)
    
    # Load metadata and split dataset
    df = load_metadata()
    dataset = split_dataset(df, val_size=0.2, random_state=SEED)
    
    # Build data loaders for training and validation
    load_train, load_validate = build_neural_loaders(
        dataset, batch_size=batch_size, num_workers=num_workers
    )
    
    # Compute class weights for the loss function
    class_weights = compute_class_weights(dataset.train_df, dataset.class_names)
    
    # Train the CNN using Lightning
    best_checkpoint = train_neural_classifier(
        epochs=100,
        batch_size=batch_size,
        learning_rate=1e-3,
        device=device,
        num_workers=num_workers,
        patience=20,
        run_name=run_name,
        train_loader=load_train,
        val_loader=load_validate,
        class_names=dataset.class_names,
        class_weights=class_weights,
        ckpt_to_start_from='runs/scratch_cnn_butterflies4/best_cnn.pt',
    )
    
    # Predict labels using the trained model and write submission file
    predict_neural_labels(
        weights_path=best_checkpoint,
        output_path=FILE_PATH / "submission_cnn.csv",
        batch_size=64,
        device=device,
    )


if __name__ == "__main__":
    main()