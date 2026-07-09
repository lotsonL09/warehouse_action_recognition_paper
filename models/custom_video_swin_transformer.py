from torchvision.models.video import swin3d_t,Swin3D_T_Weights

import torch.nn as nn  

def get_custom_video_swin_transformer(num_classes:int, phase:int):

    video_swin_model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)

    match phase:
        case 1:
            

            #Freeze weights

            for param in video_swin_model.parameters():
                param.requires_grad=False

            num_ftrs = video_swin_model.head.in_features

            custom_head = nn.Linear(num_ftrs,num_classes)

            video_swin_model.head = custom_head

            for param in video_swin_model.head.parameters():
                param.requires_grad=True

        case 2:

            #Freeze weights

            for param in video_swin_model.parameters():
                param.requires_grad=False

            num_ftrs = video_swin_model.head.in_features

            custom_head = nn.Linear(num_ftrs,num_classes)

            video_swin_model.head = custom_head

            for param in video_swin_model.head.parameters():
                param.requires_grad=True

            # unfreeze since stage 4
            for param in video_swin_model.features[5:].parameters():
                param.requires_grad=True

            for param in video_swin_model.norm.parameters():
                param.requires_grad=True
            
            for param in video_swin_model.avgpool.parameters():
                param.requires_grad=True

        case 3:

            #Freeze weights

            for param in video_swin_model.parameters():
                param.requires_grad=False

            num_ftrs = video_swin_model.head.in_features

            custom_head = nn.Linear(num_ftrs,num_classes)

            video_swin_model.head = custom_head

            for param in video_swin_model.head.parameters():
                param.requires_grad=True

            # unfreeze since stages 3 y 4
            for param in video_swin_model.features[3:].parameters():
                param.requires_grad=True

            for param in video_swin_model.norm.parameters():
                param.requires_grad=True
            
            for param in video_swin_model.avgpool.parameters():
                param.requires_grad=True
                
        case 4:

            # unfreeze all weights

            num_ftrs = video_swin_model.head.in_features

            custom_head = nn.Linear(num_ftrs,num_classes)

            video_swin_model.head = custom_head

    return video_swin_model
