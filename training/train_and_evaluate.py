import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from .training_step import train_step,train_step_triplets
from .eval_step import val_step,val_step_triplets
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR,SequentialLR

def train_and_evaluate(model: nn.Module,
                       train_dataloader: torch.utils.data.DataLoader,
                       val_dataloader: torch.utils.data.DataLoader,
                       loss_fcn: nn.Module,
                       optimizer: optim.Optimizer,
                       scheduler:optim.lr_scheduler._LRScheduler,
                       n_epochs:int,
                       save_path:Path,
                       device:torch.device,
                       ):
    
    results={
        "train_loss":[],
        "train_acc":[],
        "eval_loss":[],
        "eval_acc":[],
        "lr":[]
    }

    model=model.to(device)

    best_eval_acc=0.0


    for epoch in tqdm(range(n_epochs)):

        train_loss,train_acc=train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fcn,
                                        optimizer=optimizer,
                                        device=device)

        val_loss,val_acc=val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fcn,
                                     device=device)
        
        if isinstance(scheduler,optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if save_path != None:
            if val_acc>best_eval_acc:
                best_eval_acc = val_acc

                if scheduler != None:
                    torch.save({
                        "epoch":epoch,
                        "model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "scheduler_state_dict":scheduler.state_dict(),
                        "best_val_acc":best_eval_acc
                    }, save_path/"best_checkpoint.pth")
                else:
                    torch.save({
                        "epoch":epoch,
                        "model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "best_val_acc":best_eval_acc
                    }, save_path/"best_checkpoint.pth")

        if scheduler != None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr=optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}: Train loss: {train_loss} | Train acc: {train_acc:.4f} | LR: {current_lr}")
        print(f"\nEpoch {epoch+1}: Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["eval_loss"].append(val_loss)
        results["eval_acc"].append(val_acc)
        results["lr"].append(current_lr)
    
    del model
    del val_dataloader
    del train_dataloader

    return results


