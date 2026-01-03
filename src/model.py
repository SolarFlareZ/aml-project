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
        self.save_hyperparameters()
        
        # Loading DINO ViT-S/16 (Foundation model theta_0)
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.embed_dim = self.backbone.embed_dim 
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # Classifier head
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def train(self, mode: bool = True):
        """Forces backbone to stay in eval mode even during training if frozen."""
        super().train(mode)
        if self.hparams.freeze_backbone:
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.classifier(features)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_acc', self.train_acc(logits, y), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log('val_acc', self.val_acc(logits, y), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Crucial for sparse training: only optimize params where requires_grad=True
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            trainable_params, 
            lr=self.hparams.lr, 
            momentum=self.hparams.momentum, 
            weight_decay=self.hparams.weight_decay
        )
        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        return optimizer