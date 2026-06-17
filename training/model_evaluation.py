import torch
import torchmetrics
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

def evaluate_model(model: nn.Module,
                   model_save_path:Path,
                   val_dataloader:torch.utils.data.DataLoader,
                   num_classes:int,
                   device:str):
    
    state_dict=torch.load(str(model_save_path))
    model.load_state_dict(state_dict["model_state_dict"])

    y_preds=[]

    model.eval()

    accuracy_metric=torchmetrics.Accuracy(task="multiclass",num_classes=num_classes).to(device)
    precision_metric=torchmetrics.Precision(task="multiclass",num_classes=num_classes,average="macro").to(device)
    recall_metric=torchmetrics.Recall(task="multiclass",num_classes=num_classes,average="macro").to(device)
    f1_metric=torchmetrics.F1Score(task="multiclass",num_classes=num_classes,average="macro").to(device)

    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    with torch.inference_mode():
        for X,y in tqdm(val_dataloader,desc="Making predictions"):
            X,y=X.to(device),y.to(device)
            y_logits=model(X)

            if model.__class__.__name__ =="VivitForVideoClassification":
                y_logits=y_logits.logits

            y_pred=torch.argmax(torch.softmax(y_logits,dim=1),dim=1)
            y_preds.append(y_pred.cpu())

            accuracy_metric.update(y_pred,y)
            precision_metric.update(y_pred,y)
            recall_metric.update(y_pred,y)
            f1_metric.update(y_pred,y)

    accuracy=accuracy_metric.compute().item()
    precision=precision_metric.compute().item()
    recall=recall_metric.compute().item()
    f1=f1_metric.compute().item()

    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 score: {f1:.4f}")
