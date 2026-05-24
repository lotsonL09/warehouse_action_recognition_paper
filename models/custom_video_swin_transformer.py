from torchvision.models.video import swin3d_b, Swin3D_B_Weights

import torch.nn as nn  

def get_custom_video_swin_transformer(num_classes):

    video_swin_model = swin3d_b(weights=Swin3D_B_Weights.DEFAULT)

    #Freeze weights

    for param in video_swin_model.parameters():
        param.requires_grad=False

    num_ftrs = video_swin_model.head.in_features

    custom_head = nn.Linear(num_ftrs,num_classes)

    video_swin_model.head = custom_head

    for param in video_swin_model.head.parameters():
        param.requires_grad=True

    return video_swin_model
