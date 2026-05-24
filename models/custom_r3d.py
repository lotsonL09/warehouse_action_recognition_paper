from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn as nn

def get_custom_r3d(num_classes):
    r3d_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    #Freeze weights
    for param in r3d_model.parameters():
        param.requires_grad=False

    num_ftrs = r3d_model.fc.in_features

    custom_fc = nn.Linear(num_ftrs,num_classes)

    r3d_model.fc = custom_fc

    for param in r3d_model.fc.parameters():
        param.requires_grad=True
  
    return r3d_model