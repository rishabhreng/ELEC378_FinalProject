import pandas as pd
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    BasePredictionWriter,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.cli import LightningCLI

# import a bunch of stuff from our common module
from classify_common import (
    FILE_PATH,
    SEED,
    RUNS_DIR,
    CNN,
    ButterflyDataModule,
    get_submission_image_ids,
)


class CSVPredictionWriter(BasePredictionWriter):
    """Callback to save predictions to CSV after completion."""

    def __init__(self, output_path: str, datamodule):
        super().__init__(write_interval="batch")
        self.output_path = output_path
        self.datamodule = datamodule
        self.all_preds = []

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Collect predictions as batches complete."""
        self.all_preds.extend(predictions)

    def on_predict_end(self, trainer, pl_module):
        """Save predictions to CSV when prediction finishes."""
        if not self.all_preds:
            return

        try:
            # Ensure data module is set up
            if hasattr(self.datamodule, "setup"):
                self.datamodule.setup("predict")

            image_ids = get_submission_image_ids()
            class_names = self.datamodule.class_names

            # Convert class indices to class names
            preds = [class_names[int(idx)] for idx in self.all_preds]

            submission = pd.DataFrame({"ID": image_ids, "TARGET": preds})
            submission.to_csv(self.output_path, index=False)
            print(f"[✓] Saved {len(preds)} predictions to {self.output_path}")
        except Exception as e:
            print(f"[Error] saving predictions: {e}")


class CNNButterflyClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 100,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.01,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        # Save hyperparameters but ignore class_weights since it's a constant, not hparam
        self.save_hyperparameters(ignore=["class_weights"])
        self.cnn = CNN(num_classes=num_classes)
        self.num_classes = num_classes
        self.lr = learning_rate
        self.wd = weight_decay
        self.ls = label_smoothing

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,  # account for class imbalance
            label_smoothing=label_smoothing,
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
        self.log(
            "train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
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

    def predict_step(self, batch, batch_idx):
        x, _ = batch

        logits = self(x)
        preds = logits.argmax(dim=1)

        return preds.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )

        # Smooth LR drop with cosine fn
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=5,  # restart every 5 epochs
        #     T_mult=2,  # double the period after each restart
        # )

        # Use ReduceLROnPlateau for better plateau handling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # maximize accuracy
            factor=0.1,  # reduce LR by 10% when plateau detected
            patience=6,  # wait 3 epochs before reducing lr
            min_lr=1e-7,
        )

        # aggressive LR
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr * 10,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=0.2,
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # def on_train_epoch_end(self):
    #     current_lr = self.optimizers().param_groups[0]["lr"]
    #     self.log("learning_rate", current_lr, on_epoch=True)

    def load_state_dict(self, state_dict, strict=True):
        """Handle backward compatibility with old checkpoints that had class_weights."""
        # Remove class_weights from old checkpoints (it was a constant, not a model parameter)
        state_dict.pop("class_weights", None)
        # Remove criterion.weight from old checkpoints (weights are passed at init, not saved)
        if "criterion.weight" in state_dict:
            state_dict.pop("criterion.weight", None)
        return super().load_state_dict(state_dict, strict=strict)


class ButterflyClassifierCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """Add custom CLI arguments for run management."""
        parser.add_argument(
            "-r",
            "--run_name",
            type=str,
            default="cnn_butterflies",
            help="Name of the run directory for checkpoints and logs",
        )
        parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            default=str(FILE_PATH / "submission.csv"),
            help="Output CSV path for predictions",
        )

    def after_instantiate_classes(self):
        """Configure trainer, callbacks, and logger after classes are instantiated."""
        super().after_instantiate_classes()

        # Extract custom args from the subcommand namespace
        subcommand_config = getattr(self.config, self.config.subcommand, None)
        run_name = (
            getattr(subcommand_config, "run_name", "cnn_butterflies")
            if subcommand_config
            else "cnn_butterflies"
        )
        output_path = (
            getattr(subcommand_config, "output_path", str(FILE_PATH / "submission.csv"))
            if subcommand_config
            else str(FILE_PATH / "submission.csv")
        )

        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Configure callbacks
        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            mode="max",
            verbose=True,
        )

        checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best_cnn",
            monitor="val_accuracy",
            mode="max",
            save_top_k=3,
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
            log_weight_decay=True,
        )

        # Setup datamodule to get class names for prediction writer
        if hasattr(self.datamodule, "setup"):
            self.datamodule.setup("fit")

        pred_writer = CSVPredictionWriter(output_path, self.datamodule)

        # Set callbacks with progress bar
        self.trainer.callbacks = [
            early_stop,
            checkpoint,
            lr_monitor,
            pred_writer,
            RichProgressBar(),
        ]


def main():
    torch.set_float32_matmul_precision(
        "medium" if torch.cuda.is_available() else "high"
    )

    ButterflyClassifierCLI(
        CNNButterflyClassifier,
        ButterflyDataModule,
        seed_everything_default=SEED,
    )


if __name__ == "__main__":
    main()
