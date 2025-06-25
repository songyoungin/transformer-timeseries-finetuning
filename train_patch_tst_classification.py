"""
PatchTST Time Series Classification Training Script

This script performs time series classification using the PatchTST (Patch Time Series Transformer) model.
It uses UCR time series datasets and fine-tunes a pre-trained PatchTST model for classification tasks.

Key Features:
- Automatic loading and preprocessing of UCR datasets
- Utilization of pre-trained PatchTST models
- Efficient training through PyTorch Lightning
- Early stopping and model checkpoint functionality
- Hyperparameter configuration via command line arguments

Usage:
    # Run with default settings
    python train_patch_tst_classification.py

    # Adjust number of epochs
    python train_patch_tst_classification.py --epochs 50

    # Adjust batch size and learning rate
    python train_patch_tst_classification.py --batch-size 64 --learning-rate 2e-4

    # Adjust all parameters
    python train_patch_tst_classification.py --epochs 200 --batch-size 16 --learning-rate 1e-5 --val-split 0.3

    # Show help
    python train_patch_tst_classification.py --help
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig, PatchTSTForClassification
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sktime.datasets import load_UCR_UEA_dataset
from typing import Optional
import warnings
from torchmetrics import Accuracy
import argparse

warnings.filterwarnings("ignore")


class TimeSeriesDataModule(pl.LightningDataModule):
    """Lightning data module for processing time series classification data."""

    def __init__(
        self,
        dataset_name: str = "GunPoint",
        batch_size: int = 16,
        val_split: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize the TimeSeriesDataModule.

        Args:
            dataset_name: Name of the UCR dataset.
            batch_size: Batch size for data loaders.
            val_split: Validation data ratio.
            random_state: Random seed for reproducibility.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        """Download data if necessary."""
        # Automatically downloaded by sktime
        pass

    def setup(self, stage: Optional[str] = None):
        """Load and preprocess data."""
        # Load UCR dataset
        X_train, y_train = load_UCR_UEA_dataset(
            name=self.dataset_name, split="train", return_X_y=True
        )
        X_test, y_test = load_UCR_UEA_dataset(
            name=self.dataset_name, split="test", return_X_y=True
        )

        # Convert data (pandas DataFrame to numpy array)
        X_train = self._convert_to_numpy(X_train)
        X_test = self._convert_to_numpy(X_test)

        # Label encoding
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Data normalization
        X_train_scaled = self._scale_data(X_train, fit=True)
        X_test_scaled = self._scale_data(X_test, fit=False)

        # Split training data into train/validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled,
            y_train_encoded,
            test_size=self.val_split,
            random_state=self.random_state,
            stratify=y_train_encoded,
        )

        # Convert to tensors
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train_split), torch.LongTensor(y_train_split)
        )
        self.val_dataset = TensorDataset(
            torch.FloatTensor(X_val_split), torch.LongTensor(y_val_split)
        )
        self.test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test_encoded)
        )

        # Store data attributes
        self.num_classes = len(self.label_encoder.classes_)
        self.seq_len = X_train_scaled.shape[1]
        self.num_features = X_train_scaled.shape[2]

    def _convert_to_numpy(self, X) -> np.ndarray:
        """Convert pandas DataFrame to numpy array.

        Args:
            X: Input data in pandas DataFrame format.

        Returns:
            Converted numpy array with shape (samples, time_steps, features).
        """
        # Handle sktime UCR data format
        if hasattr(X, "values"):
            # For pandas DataFrame
            X_list = []
            for i in range(len(X)):
                # Extract time series data from each row
                series = X.iloc[i, 0]  # Time series data is in the first column
                if hasattr(series, "values"):
                    X_list.append(series.values)
                else:
                    X_list.append(np.array(series))
            X_array = np.array(X_list)
        else:
            X_array = np.array(X)

        # Convert to 3D array (samples, time_steps, features)
        if X_array.ndim == 2:
            X_array = X_array.reshape(X_array.shape[0], X_array.shape[1], 1)

        return X_array

    def _scale_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize time series data.

        Args:
            X: Input time series data.
            fit: Whether to fit the scaler on the data.

        Returns:
            Normalized time series data.
        """
        # Normalize entire data as one batch
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])  # (samples * timesteps, features)

        if fit:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)

        return X_scaled.reshape(original_shape)

    def train_dataloader(self):
        """Create training data loader.

        Returns:
            DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation data loader.

        Returns:
            DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test data loader.

        Returns:
            DataLoader for test data.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


class PatchTSTClassificationModule(pl.LightningModule):
    """Lightning module for PatchTST classification model."""

    def __init__(
        self,
        pretrained_name: str = "namctin/patchtst_etth1_pretrain",
        num_classes: int = 2,
        seq_len: int = 150,
        num_features: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """Initialize the PatchTSTClassificationModule.

        Args:
            pretrained_name: Name of the pretrained model.
            num_classes: Number of classes for classification.
            seq_len: Sequence length.
            num_features: Number of input features.
            learning_rate: Learning rate for optimization.
            weight_decay: Weight decay for regularization.
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Model configuration
        config = PatchTSTConfig.from_pretrained(pretrained_name)
        config.num_targets = num_classes
        config.use_cls_token = True
        config.context_length = seq_len
        config.num_input_channels = num_features

        # Model initialization
        self.model = PatchTSTForClassification.from_pretrained(
            pretrained_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics storage
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        return self.model(past_values=x)

    def training_step(self, batch, batch_idx):
        """Perform a training step.

        Args:
            batch: Batch of training data.
            batch_idx: Index of the current batch.

        Returns:
            Training loss.
        """
        x, y = batch
        outputs = self(x)

        # Calculate loss
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.prediction_logits
            loss = self.criterion(logits, y)

        # Calculate accuracy
        logits = outputs.prediction_logits
        preds = torch.argmax(logits, dim=-1)
        acc = self.train_accuracy(preds, y)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step.

        Args:
            batch: Batch of validation data.
            batch_idx: Index of the current batch.

        Returns:
            Validation loss.
        """
        x, y = batch
        outputs = self(x)

        # Calculate loss
        logits = outputs.prediction_logits
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = self.val_accuracy(preds, y)

        # Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Perform a test step.

        Args:
            batch: Batch of test data.
            batch_idx: Index of the current batch.

        Returns:
            Test loss.
        """
        x, y = batch
        outputs = self(x)

        # Calculate loss
        logits = outputs.prediction_logits
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = self.test_accuracy(preds, y)

        # Logging
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary containing optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train PatchTST for time series classification"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimization (default: 1e-4)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed
    pl.seed_everything(42)

    # Initialize data module
    print("Loading data...")
    data_module = TimeSeriesDataModule(
        dataset_name="GunPoint",
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    data_module.setup()

    print("Dataset information:")
    print(f"- Number of classes: {data_module.num_classes}")
    print(f"- Sequence length: {data_module.seq_len}")
    print(f"- Number of features: {data_module.num_features}")
    print(f"- Training samples: {len(data_module.train_dataset)}")
    print(f"- Validation samples: {len(data_module.val_dataset)}")
    print(f"- Test samples: {len(data_module.test_dataset)}")

    print("\nTraining configuration:")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.learning_rate}")
    print("- Dataset: GunPoint")
    print(f"- Validation split: {args.val_split}")

    # Initialize model
    model = PatchTSTClassificationModule(
        num_classes=data_module.num_classes,
        seq_len=data_module.seq_len,
        num_features=data_module.num_features,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
    )

    # Callback setup
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min"),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            filename="best_model_{epoch:02d}_{val_acc:.4f}",
        ),
    ]

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=16,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    # Start training
    print("\nStarting training...")
    trainer.fit(model, data_module)

    # Start testing
    print("Starting testing...")
    trainer.test(model, data_module, ckpt_path="best")

    # Final results
    print("Training completed!")


if __name__ == "__main__":
    main()
