from typing import Literal

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


class DinoClassifier(pl.LightningModule):    
    def __init__(
        self,
        num_classes: int = 100,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        scheduler: Literal['cosine', 'step', 'none'] = 'cosine',
        max_epochs: int = 100,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        # args are used through self.hparams, they are not unused so don't remove them
        self.save_hyperparameters()
        
        # loading dino
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            # double check that this freezes completely
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): # this does not consider freeze=false, need fix
            features = self.backbone(x)
        return self.classifier(features)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_acc(logits, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.SGD(
            params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        
        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
            )
        elif self.hparams.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        else:
            return optimizer
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        } # read docs, if no scheduler, only return optimizer