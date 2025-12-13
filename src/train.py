import os

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from .datamodule import CIFAR100DataModule
from .model import DinoClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:    
    print("config:")
    print(OmegaConf.to_yaml(cfg))
    
    # hydra chagnes root dir
    original_cwd = get_original_cwd()
    data_dir = os.path.join(original_cwd, cfg.data.data_dir)
    checkpoint_dir = os.path.join(original_cwd, cfg.callbacks.model_checkpoint.dirpath)
    log_dir = os.path.join(original_cwd, cfg.logging.log_dir)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    pl.seed_everything(cfg.seed, workers=True)
    
    datamodule = CIFAR100DataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        seed=cfg.seed,
        image_size=cfg.model.image_size
    )
    
    model = DinoClassifier(
        num_classes=cfg.model.num_classes,
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        scheduler=cfg.scheduler.name,
        max_epochs=cfg.trainer.max_epochs,
        freeze_backbone=cfg.model.freeze_backbone
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.callbacks.model_checkpoint.monitor,
            mode=cfg.callbacks.model_checkpoint.mode,
            save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
            dirpath=checkpoint_dir,
            filename=cfg.callbacks.model_checkpoint.filename,
            save_last=True
        ),
        EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    

    logger = CSVLogger(log_dir, name=cfg.experiment_name)
    
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger
    )
    
    # checking checkpoints
    ckpt_path = cfg.get('resume_from', None)
    if ckpt_path and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(original_cwd, ckpt_path)
    
    print("starting training...")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    
    print("starting testing...")
    trainer.test(model, datamodule, ckpt_path='best')
    test_acc = trainer.callback_metrics.get('test_acc', None)

    
    print("training complete")
    print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"best val_acc: {trainer.checkpoint_callback.best_model_score:.4f}")
    print(f"best test_acc: {test_acc:.4f}")

if __name__ == "__main__":
    train()