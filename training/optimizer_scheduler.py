import torch.nn as nn
import torch.optim as optim
import math
from config.train_config import SchedulerClass

'''
YA NO SE VA A USAR
'''

def define_optimizer_and_scheduler(model:nn.Module,
                                   learning_rate:float,
                                   weight_decay:float,
                                   warmup_epochs:int,
                                   total_epochs:int,
                                   train_datalaoder=None,
                                   scheduler:SchedulerClass=None):
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if scheduler != None:

        if scheduler.name == "reduceLrOnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler.reduceLrOnPlateau.mode,
                factor=scheduler.reduceLrOnPlateau.factor,
                patience=scheduler.reduceLrOnPlateau.patience
            )
        elif scheduler.name == "stepLr":
            scheduler=optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler.stepLr.step_size,
                gamma=scheduler.stepLr.gamma #reduce the learning rate by 20%
            ) 
        elif scheduler.name == "cosineAnnealingLr":
            scheduler=optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler.cosineAnnealingLr.t_max,
                eta_min=scheduler.cosineAnnealingLr.eta_min
            )
        elif scheduler.name == "warmup-cosine-decay":

            steps=len(train_datalaoder)
            
            total_steps=total_epochs*steps
            warmup_steps=warmup_epochs*steps

            def lr_lambda(current_step):
                #  Warmup
                if current_step < warmup_steps:
                    return float(current_step +1) / float(max(1, warmup_steps))

                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))

                min_factor = 0.01

                return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = optim.lr_scheduler.LambdaLR(
                                                    optimizer,
                                                    lr_lambda=lr_lambda
                                                )

    return optimizer,scheduler