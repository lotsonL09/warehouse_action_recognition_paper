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
                       scheduler_name:str,
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

        scheduler_step_per_batch = True if scheduler_name == "warmup-cosine-decay" else False

        train_loss,train_acc=train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fcn,
                                        optimizer=optimizer,
                                        device=device,
                                        scheduler=scheduler,
                                        scheduler_step_per_batch=scheduler_step_per_batch)

        val_loss,val_acc=val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fcn,
                                     device=device)
        
        if scheduler != None and not scheduler_step_per_batch:
            if isinstance(scheduler,optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
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
    
    return results

def train_and_evaluate_triplets(model: nn.Module,
                       train_dataloader: torch.utils.data.DataLoader,
                       val_dataloader: torch.utils.data.DataLoader,
                       emb_val_dataloader:torch.utils.data.DataLoader,
                       loss_fcn: nn.Module,
                       optimizer: optim.Optimizer,
                       scheduler_name:str,
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
        "lr":[],
        "r1":[],
        "r5":[]
    }

    model=model.to(device)

    best_eval_loss=float("inf")


    for epoch in tqdm(range(n_epochs)):

        scheduler_step_per_batch = True if scheduler_name == "warmup-cosine-decay" else False

        train_loss,train_acc=train_step_triplets(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fcn,
                                        optimizer=optimizer,
                                        device=device,
                                        scheduler=scheduler,
                                        scheduler_step_per_batch=scheduler_step_per_batch)

        val_loss,val_acc=val_step_triplets(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fcn,
                                     device=device)
        
        
        if (epoch+1)%5==0:
            embeddings,labels=get_embeddings(model=model,
                       dataloader=emb_val_dataloader,
                       device=device)
            r1=recall_at_k(embeddings,labels,K=1)
            r5=recall_at_k(embeddings,labels,K=5)
            results["r1"].append(r1)
            results["r5"].append(r5)
            print(f"\nRecall@1: {r1:.4f} | Recall@5: {r5:.4f}")

        
        if scheduler != None and not scheduler_step_per_batch:
            if isinstance(scheduler,optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if save_path != None:
            if val_loss<best_eval_loss:
                best_eval_loss = val_loss

                if scheduler != None:
                    torch.save({
                        "epoch":epoch,
                        "model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "scheduler_state_dict":scheduler.state_dict(),
                        "best_val_acc":best_eval_loss
                    }, save_path/"best_checkpoint.pth")
                else:
                    torch.save({
                        "epoch":epoch,
                        "model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "best_val_acc":best_eval_loss
                    }, save_path/"best_checkpoint.pth")

        if scheduler != None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr=optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}: Train loss: {train_loss:.4f} | Train acc: {train_acc} | LR: {current_lr}")
        print(f"\nEpoch {epoch+1}: Val loss: {val_loss:.4f} | Val acc:{val_acc}")

        results["train_loss"].append(train_loss)
        results["eval_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["eval_acc"].append(val_acc)
        results["lr"].append(current_lr)
    
    return results

def get_embeddings(model, dataloader, device):
    model.eval()
    
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            emb = model(x)

            all_embeddings.append(emb.cpu())
            all_labels.append(y)

    return torch.cat(all_embeddings), torch.cat(all_labels)

def recall_at_k(embeddings, labels, K=5):
    
    # Matriz de distancias (NxN)
    distances = torch.cdist(embeddings, embeddings)

    # Para no compararse consigo mismo
    diag_mask = torch.eye(distances.size(0), dtype=torch.bool)
    distances[diag_mask] = float('inf')

    # Obtener K vecinos más cercanos
    knn_indices = distances.topk(K, largest=False).indices

    correct = 0

    for i in range(embeddings.size(0)):
        anchor_label = labels[i]
        neighbor_labels = labels[knn_indices[i]]

        # ¿alguno coincide?
        if (neighbor_labels == anchor_label).any():
            correct += 1

    return correct / embeddings.size(0)

