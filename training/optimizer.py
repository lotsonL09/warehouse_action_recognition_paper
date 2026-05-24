from torch import optim
from config.optimizer_config import OptimizerConfig
import torch.nn as nn


'''
*VivitForVideoClassification: base learning rate 0.1 | no weight decay | momentum 0.9

*S3D: base learning rate 0.1 | momentum 0.9

*MViT: base learning rate 1.6 x 10^-3 | weight decay 5 x 10^-2 | label smoothing 0.1 - Loss function

*VideoResNet: base learning rate 0.1 | weight decay 1 x 10^-3 | momentum 0.9

    The learning rate is decayed by a factor of 10, using ReduceLronPlateau, if the validation loss does not improve for 3 epochs.

*VideoSwinTransformer: weight decay 5 x 10^-2 

    The learning rate of the backbone architecture needs to be smaller (x0.1) than that of the head.
    3e-5 -> backbone, 3e-4 -> head

'''

def build_optimizer(model:nn.Module,
                    optimizer_config=OptimizerConfig):
    
    match model.__class__.__name__:
        case "VivitForVideoClassification" | "S3D":
            optimizer = optim.SGD(
                params=model.parameters(),
                lr=optimizer_config.lr,
                momentum=optimizer_config.momentum
            )
        case "MViT":
            optimizer = optim.AdamW(
                params=model.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay
            )
        case "VideoResNet":
            optimizer=optim.SGD(
                params=model.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                momentum=optimizer_config.momentum
            )
        case "SwinTransformer3d":

            backbone=nn.ModuleList([
                model.patch_embed,
                model.features,
                model.norm
            ])

            optimizer = optim.AdamW(
                params=[
                    {"params": backbone.parameters(), "lr": optimizer_config.lr * 0.1},
                    {"params": model.head.parameters(), "lr": optimizer_config.lr}
                ],
                weight_decay=optimizer_config.weight_decay)

    return optimizer

