import torch
import torch.nn as nn

def train_step(model:torch.nn.Module,
            dataloader:torch.utils.data.dataloader,
            loss_fn:torch.nn.Module,
            optimizer:torch.optim.Optimizer,
            device,
            scheduler=None):

    model.train()

    total_loss=0.0

    total_correct=0

    total_samples=0

    for batch, (X,y) in enumerate(dataloader):
        
        X,y=X.to(device),y.to(device)

        optimizer.zero_grad()

        outputs=model(X)

        loss=loss_fn(outputs,y)

        batch_size=y.size(0)

        total_loss+=loss.item()*batch_size

        loss.backward()

        optimizer.step()

        _,predicted=outputs.max(1)

        total_correct+= (predicted == y).sum().item()

        total_samples+=batch_size

    train_loss= total_loss/total_samples
    
    train_acc=total_correct/total_samples

    return train_loss,train_acc



def train_step_triplets(model:torch.nn.Module,
                        dataloader:torch.utils.data.dataloader,
                        loss_fn:torch.nn.Module,
                        optimizer:torch.optim.Optimizer,
                        device,
                        scheduler=None,
                        scheduler_step_per_batch=False):

    model.train()

    total_loss=0.0

    total_samples=0

    correct=0

    for _, (anchors,positives,negatives,_) in enumerate(dataloader):
        
        anchors,positives,negatives=anchors.to(device),positives.to(device),negatives.to(device)

        optimizer.zero_grad()

        anchor_embs=model(anchors)
        positive_embs=model(positives)
        negative_embs=model(negatives)

        loss=loss_fn(anchor_embs,positive_embs,negative_embs)

        loss.backward()

        optimizer.step()

        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        samples=anchors.size(0)

        d_ap = nn.functional.pairwise_distance(anchor_embs, positive_embs)
        d_an = nn.functional.pairwise_distance(anchor_embs, negative_embs)

        correct += (d_ap < d_an).sum().item()

        total_loss+=loss.item()*samples

        total_samples+=samples

    train_loss= total_loss/total_samples
    train_acc=correct/total_samples
    
    return train_loss,train_acc