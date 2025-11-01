# modules.py
from __future__ import annotations
from typing import Iterable, Optional
import torch
import torch.nn as nn
import timm

__all__ = [
    "ConvNeXtClassifier",
    "build_model",
    "freeze_backbone",
    "unfreeze_all",
    "num_trainable_params",
]

class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    model_name: str = "convnext_tiny",
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    return ConvNeXtClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )



def freeze_backbone(model: nn.Module, except_head: bool = True) -> None:
    """
    Freeze parameters for finetuning. If except_head=True, keep classifier head trainable.
    Works with common timm naming (e.g., model.backbone.head).
    """
    
    for p in model.parameters():
        p.requires_grad = False

    if except_head:
        
        head = None
        if hasattr(model, "backbone") and hasattr(model.backbone, "get_classifier"):
            
            head = model.backbone.get_classifier()
        elif hasattr(model, "backbone") and hasattr(model.backbone, "head"):
            head = model.backbone.head

        if head is None:
           
            for m in model.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    head = m  
            
            if head is None:
                for p in model.parameters():
                    p.requires_grad = True
                return

        for p in head.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
