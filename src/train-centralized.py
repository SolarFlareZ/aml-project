import os
import sys
import torch
sys.path.append(os.getcwd())
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from src.datamodule import CIFAR100DataModule
from src.model import DinoClassifier

def train_centralized():
    # 1. Setup Data
    # For centralized training, we don't care about fl_clients yet
    dm = CIFAR100DataModule(
        data_dir='./data',
        batch_size=128,
        image_size=224,
        val_split=0.1
    )

    # 2. Setup Model
    # freeze_backbone=True for the initial baseline (Linear Probing)
    model = DinoClassifier(
        num_classes=100,
        lr=0.01,
        scheduler='cosine',
        max_epochs=5, # Start with 20 to get results quickly on laptop
        freeze_backbone=True 
    )

    # 3. Setup Logger and Callbacks
    # This creates a 'logs' folder with CSVs you can use for plotting in Excel/Matplotlib
    logger = CSVLogger("logs", name="centralized_baseline")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="dino-cifar100-baseline",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    # 4. Trainer
    # 'accelerator="auto"' will use your GPU if available, else CPU
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        precision=16 if torch.cuda.is_available() else 32 # Mixed precision for speed
    )

    # 5. Start Training
    print("Starting Centralized Baseline Training...")
    trainer.fit(model, datamodule=dm)
    
    # 6. Test
    print("Evaluating on Test Set...")
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    train_centralized()