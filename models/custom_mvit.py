from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
import torch.nn as nn

def get_custom_mvit(num_classes:int,phase:int):

    match phase:
        case 1:
            mvit_model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

            #Freeze weights
            for param in mvit_model.parameters():
                param.requires_grad=False

            num_ftrs = mvit_model.head[1].in_features

            fc_custom = nn.Linear(num_ftrs,num_classes)

            mvit_model.head[1] = fc_custom

            for param in mvit_model.head.parameters():
                param.requires_grad=True
        case 2:

            # unfreeze all weights
            mvit_model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

            num_ftrs = mvit_model.head[1].in_features

            fc_custom = nn.Linear(num_ftrs,num_classes)

            mvit_model.head[1] = fc_custom
        case 3:
            mvit_model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

            for param in mvit_model.parameters():
                param.requires_grad=False

            num_ftrs = mvit_model.head[1].in_features

            fc_custom = nn.Linear(num_ftrs,num_classes)

            mvit_model.head[1] = fc_custom

            for param in mvit_model.head.parameters():
                param.requires_grad=True

            #unfreeze stage 4
            for param in mvit_model.blocks[14:].parameters():
                param.requires_grad=True

        case 4:
            mvit_model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

            for param in mvit_model.parameters():
                param.requires_grad=False

            num_ftrs = mvit_model.head[1].in_features

            fc_custom = nn.Linear(num_ftrs,num_classes)

            mvit_model.head[1] = fc_custom

            for param in mvit_model.head.parameters():
                param.requires_grad=True

            #unfreeze stages  3 y 4
            for param in mvit_model.blocks[3:].parameters():
                param.requires_grad=True
        case 5:
            pass


    return mvit_model
