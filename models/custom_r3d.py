from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn as nn

def get_custom_r3d(num_classes:int,phase:int):

    match phase:
        case 1:
            
            r3d_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

            #Freeze weights
            for param in r3d_model.parameters():
                param.requires_grad=False

            num_ftrs = r3d_model.fc.in_features

            custom_fc = nn.Linear(num_ftrs,num_classes)

            r3d_model.fc = custom_fc

            for param in r3d_model.fc.parameters():
                param.requires_grad=True

        case 2:
            # unfreeze all weights

            r3d_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

            num_ftrs = r3d_model.fc.in_features

            custom_fc = nn.Linear(num_ftrs,num_classes)

            r3d_model.fc = custom_fc

        case 3:
            r3d_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

            #Freeze weights
            for param in r3d_model.parameters():
                param.requires_grad=False

            num_ftrs = r3d_model.fc.in_features

            custom_fc = nn.Linear(num_ftrs,num_classes)

            r3d_model.fc = custom_fc

            for param in r3d_model.fc.parameters():
                param.requires_grad=True

            for param in r3d_model.avgpool.parameters():
                param.requires_grad=True
            
            for param in r3d_model.layer4.parameters():
                param.requires_grad=True
            
        case 4:
            r3d_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

            #Freeze weights
            for param in r3d_model.parameters():
                param.requires_grad=False

            num_ftrs = r3d_model.fc.in_features

            custom_fc = nn.Linear(num_ftrs,num_classes)

            r3d_model.fc = custom_fc

            for param in r3d_model.fc.parameters():
                param.requires_grad=True

            for param in r3d_model.avgpool.parameters():
                param.requires_grad=True
            
            for param in r3d_model.layer4.parameters():
                param.requires_grad=True
            
            for param in r3d_model.layer3.parameters():
                param.requires_grad=True
            
        case 5:
            pass
  
    return r3d_model