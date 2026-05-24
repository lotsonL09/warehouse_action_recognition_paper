import torch.nn as nn
from torch import optim
from config.scheduler_config import SchedulerConfig
from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingLR, 
                                      SequentialLR, ReduceLROnPlateau,
                                      StepLR)

'''
para las primeras pruebas que los warmup epochs sean el 10% del total.
'''

def build_scheduler(model:nn.Module,
                    optimizer:optim.Optimizer,
                    scheduler_config:SchedulerConfig,
                    total_epochs:int,
                    warmup_epochs:int|None = None
                    ):

    match model.__class__.__name__: 
        case "VivitForVideoClassification" | "SwinTransformer3d": # done
            warmup_scheduler= LinearLR(
                optimizer,
                start_factor=scheduler_config.linear_lr.start_factor,
                end_factor=scheduler_config.linear_lr.end_factor,
                total_iters=warmup_epochs
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=scheduler_config.cosine_annealing_lr.eta_min
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        case "S3D": # done - probar - luego probar con algo mas cercano al paper
            scheduler = StepLR(
                optimizer,
                step_size=total_epochs//scheduler_config.step_lr.factor,
                gamma=scheduler_config.step_lr.gamma
            )
        case "MViT": # done - probar - luego probar con algo mas cercano al paper
            warmup_scheduler= LinearLR(
                optimizer,
                start_factor=scheduler_config.linear_lr.start_factor,
                end_factor=scheduler_config.linear_lr.end_factor,
                total_iters=warmup_epochs
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=scheduler_config.cosine_annealing_lr.eta_min
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        case "VideoResNet": # validation loss | done
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.reduce_lr_on_plateau.mode,
                factor=scheduler_config.reduce_lr_on_plateau.factor,
                patience=scheduler_config.reduce_lr_on_plateau.patience
            )
        
    return scheduler