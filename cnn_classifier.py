# import a bunch of stuff
import pandas as pd
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import argparse

# import a bunch of stuff from our common module
from classify_common import (
    FILE_PATH,
    SEED,
    RUNS_DIR,
    ConvNeurNetwork,
    ButterflyDataModule,
    get_submission_image_ids,
    set_seed,
)

class CNNButterflyClassifier(L.LightningModule):    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.01,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.cnn = ConvNeurNetwork(num_classes=num_classes)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        
        # add class weights to normalize for class imbalance
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        
        # use cross entropy loss with class weights and optional label smoothing
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
        
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch

        logits = self(x)
        preds = logits.argmax(dim=1)

        return preds.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Smooth LR drop with cosine fn
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,  # restart every 5 epochs
            T_mult=2,  # double the period after each restart
        )

        # Use ReduceLROnPlateau for better plateau handling
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',  # maximize accuracy
        #     factor=0.1,  # reduce LR by 10% when plateau detected
        #     patience=5,  # wait 5 epochs before reducing
        #     min_lr=1e-7,
        # )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
            }
        }

    # Log the learning rate at the end of each training epoch for monitoring
    def on_train_epoch_end(self):
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_epoch=True)

def main():
    parser = argparse.ArgumentParser(  
        description='CNN Butterfly/Moth Species Classifier Training/Prediction Script')  
    parser.add_argument('-r', '--run_name', type=str, default="cnn_butterflies")
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-d', '--device', type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('-n', '--num-workers', type=int, default=4)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-ls', '--label_smoothing', type=float, default=0.01)
    parser.add_argument('-vs', '--val_size', type=float, default=0.2, help="Proportion of training data to use for validation (between 0 and 1)")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to a checkpoint to start training from / predict with (optional)")
    parser.add_argument('--output_path', type=str, default=FILE_PATH / "submission.csv", help="Path to save the submission CSV file with predictions")
    parser.add_argument('--no_train', action='store_true', help="Skip training and only run prediction using the best checkpoint from the specified run directory")
    parser.add_argument('--no_predict', action='store_true', help="Skip prediction and only run training")

    args = parser.parse_args() 

    torch.set_float32_matmul_precision('medium' if torch.cuda.is_available() else 'high')
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # Set the random seed for reproducibility
    set_seed(SEED)
    
    # Create the data module
    data_module = ButterflyDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        seed=SEED,
    )
    
    # Setup the data module to prepare data
    if not args.no_train:
        data_module.setup(stage="fit")
        num_classes = len(data_module.class_names)
        class_weights = data_module.class_weights
    else:
        num_classes = None
        class_weights = None

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint to get num_classes if not training
    if num_classes is None and args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        hparams = checkpoint.get("hyper_parameters", {})
        num_classes = hparams.get("num_classes", 100)
    elif num_classes is None:
        raise ValueError("Must provide num_classes (either through training data or checkpoint)")
    
    model = CNNButterflyClassifier(
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
    )
    
    # Define callbacks for early stopping and checkpointing
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=args.patience,
        mode="max",
    )
    
    checkpointer = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_cnn",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    
    # Create the Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=device if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[early_stop, checkpointer, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    
    if not args.no_train:
        # Train the model
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=args.ckpt_path,
        )

    if not args.no_predict:
        # Setup data module for prediction
        data_module.setup(stage="predict")
        
        # Determine which checkpoint to use
        ckpt_to_use = checkpointer.best_model_path if not args.no_train else args.ckpt_path
        
        # Load model from checkpoint with strict=False to ignore old state_dict keys
        model = CNNButterflyClassifier.load_from_checkpoint(
            ckpt_to_use,
            num_classes=model.cnn.classifier[-1].out_features,
            strict=False
        )
        
        # Run predictions (no ckpt_path since model is already loaded with checkpoint)
        preds = trainer.predict(
            model=model,
            datamodule=data_module,
        )
        
        image_ids = get_submission_image_ids()
        # Reverse preds to original string labels
        preds = [data_module.class_names[pred[0]] for pred in preds]

        # Generate submission CSV
        submission = pd.DataFrame({"ID": image_ids, "TARGET": preds})
        submission.to_csv(args.output_path, index=False)
        print(f"Wrote submission file to {args.output_path}")
    

if __name__ == "__main__":
    main()