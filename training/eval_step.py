import torch
import torch.nn as nn

def val_step(model:torch.nn.Module,
            dataloader:torch.utils.data.dataloader,
            loss_fn:torch.nn.Module,
            device):

    model.eval()

    total_loss=0.0

    total_correct,total_samples=0,0

    with torch.inference_mode():

        for batch, (X,y) in enumerate(dataloader):

            X,y=X.to(device),y.to(device)

            outputs=model(X)

            if model.__class__.__name__ == "VivitForVideoClassification":
                loss = loss_fn(outputs.logits, y)
            else:
                loss=loss_fn(outputs,y)

            batch_size=y.size(0)

            total_loss+=loss.item()*batch_size

            if model.__class__.__name__ == "VivitForVideoClassification":
                _,predicted_indices=outputs.logits.max(1)
            else:
                _,predicted_indices=outputs.max(1)        
            
            total_correct += (predicted_indices == y).sum().item()

            total_samples+=batch_size

    val_loss=total_loss / total_samples

    val_acc=total_correct/total_samples

    return val_loss,val_acc

def val_step_triplets(model:torch.nn.Module,
                    dataloader:torch.utils.data.dataloader,
                    loss_fn:torch.nn.Module,
                    device):

    model.eval()

    total_loss=0.0

    correct,total_samples=0,0

    with torch.inference_mode():

        for _, (anchors,positives,negatives,_) in enumerate(dataloader):

            anchors,positives,negatives=anchors.to(device),positives.to(device),negatives.to(device)

            anchor_embs=model(anchors)
            positive_embs=model(positives)
            negative_embs=model(negatives)

            loss=loss_fn(anchor_embs,positive_embs,negative_embs)

            d_ap = nn.functional.pairwise_distance(anchor_embs, positive_embs)
            d_an = nn.functional.pairwise_distance(anchor_embs, negative_embs)

            correct += (d_ap < d_an).sum().item()

            samples= anchors.size(0)

            total_loss+=loss.item()*samples

            total_samples+=samples

    val_loss=total_loss / total_samples
    val_acc=correct/total_samples

    return val_loss,val_acc