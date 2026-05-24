from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
import torch.nn as nn

def get_custom_mvit(num_classes):
    mvit_model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

    #Freeze weights
    for param in mvit_model.parameters():
        param.requires_grad=False

    num_ftrs = mvit_model.head[1].in_features

    fc_custom = nn.Linear(num_ftrs,num_classes)

    mvit_model.head[1] = fc_custom

    for param in mvit_model.head.parameters():
        param.requires_grad=True

    return mvit_model
