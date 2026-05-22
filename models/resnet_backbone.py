import torch
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights


def custom_resnet(embedding_dim):
    model_res=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    conv1_old_weights=model_res.conv1.weight.data
    model_res.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False)
    with torch.no_grad():
        model_res.conv1.weight.copy_(conv1_old_weights[:, :, 2:5, 2:5])
    model_res.maxpool=nn.Identity()

    num_ftrs=model_res.fc.in_features
    model_res.fc=nn.Linear(num_ftrs,embedding_dim)

    for param in model_res.parameters():
        param.requires_grad=False
    
    #Descongelamos desde la capa 4
    for param in model_res.layer4.parameters():
        param.requires_grad=True

    for param in model_res.avgpool.parameters():
        param.requires_grad=True

    for param in model_res.fc.parameters():
        param.requires_grad=True

    return model_res