"""
Generate predictions on validation set to analyze which classes the model gets wrong.
"""
import pandas as pd
import torch
import lightning as L
import argparse
from pathlib import Path

from classify_common import (
    FILE_PATH,
    SEED,
    RUNS_DIR,
    ButterflyDataModule,
    set_seed,
)
from cnn_classifier import CNNButterflyClassifier

def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions on validation set')
    parser.add_argument('-r', '--run_name', type=str, default="scratch_cnn_butterflies5",
                        help="Run directory name")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-d', '--device', type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('-n', '--num-workers', type=int, default=4)
    parser.add_argument('-vs', '--val_size', type=float, default=0.2,
                        help="Proportion of training data to use for validation")
    parser.add_argument('-o', '--output', type=str, default="val_predictions.csv",
                        help="Output CSV filename")
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium' if torch.cuda.is_available() else 'high')
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    
    set_seed(SEED)
    
    # Create and setup data module
    data_module = ButterflyDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        seed=SEED,
    )
    data_module.setup(stage="fit")
    
    # Get validation dataframe and class names
    val_df = data_module.val_df.copy()
    class_names = data_module.class_names
    num_classes = len(class_names)
    
    # Find best checkpoint
    run_dir = RUNS_DIR / args.run_name
    ckpt_path = run_dir / "best_cnn.ckpt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # Load model from checkpoint
    model = CNNButterflyClassifier.load_from_checkpoint(
        str(ckpt_path),
        num_classes=num_classes,
        strict=False
    )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator=device if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        enable_progress_bar=True,
    )
    
    # Run predictions on validation set
    print("Running predictions on validation set...")
    preds = trainer.predict(
        model=model,
        dataloaders=data_module.val_dataloader(),
    )
    
    # Flatten predictions
    all_preds = []
    for batch_preds in preds:
        all_preds.extend(batch_preds)
    
    # Convert predictions from indices to class names
    pred_names = [class_names[pred] for pred in all_preds]
    true_names = val_df["TARGET"].tolist()
    
    # Create result dataframe
    results = pd.DataFrame({
        "file_name": val_df["file_name"].tolist(),
        "true_class": true_names,
        "predicted_class": pred_names,
        "correct": [t == p for t, p in zip(true_names, pred_names)],
    })
    
    # Save to CSV
    results.to_csv(args.output, index=False)
    print(f"\nSaved validation predictions to {args.output}")
    
    # Print summary statistics
    correct_count = results["correct"].sum()
    total_count = len(results)
    accuracy = correct_count / total_count * 100
    
    print(f"\nValidation Summary:")
    print(f"Total images: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Print per-class accuracy
    print(f"\nPer-class accuracy:")
    for class_name in class_names:
        class_mask = results["true_class"] == class_name
        if class_mask.sum() > 0:
            class_acc = results[class_mask]["correct"].sum() / class_mask.sum() * 100
            class_count = class_mask.sum()
            print(f"  {class_name}: {class_acc:.1f}% ({int(results[class_mask]['correct'].sum())}/{int(class_count)})")
    
    # Print most common misclassifications
    print(f"\nMost common misclassifications:")
    wrong = results[~results["correct"]]
    if len(wrong) > 0:
        misclass = (wrong["true_class"] + " -> " + wrong["predicted_class"]).value_counts().head(10)
        for i, (name, count) in enumerate(misclass.items(), 1):
            print(f"  {i}. {name}: {count} times")

if __name__ == "__main__":
    main()
