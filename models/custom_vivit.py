from transformers import VivitForVideoClassification, VivitConfig

import torch.nn as nn

def get_custom_vivit(num_classes:int,phase:int):

    match phase:
        case 1:

            config=VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")

            config.num_frames=16

            vivit_model=VivitForVideoClassification(config)

            #Freeze weights
            for param in vivit_model.parameters():
                param.requires_grad=False

            num_ftrs = vivit_model.classifier.in_features

            custom_classifier = nn.Linear(num_ftrs,num_classes)

            vivit_model.classifier = custom_classifier

            for param in vivit_model.classifier.parameters():
                param.requires_grad=True
        case 2:

            config=VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")

            config.num_frames=16

            vivit_model=VivitForVideoClassification(config)

            num_ftrs = vivit_model.classifier.in_features

            custom_classifier = nn.Linear(num_ftrs,num_classes)

            vivit_model.classifier = custom_classifier

        case 3:
            pass
        case 4:
            pass
        case 5:
            pass

    return vivit_model 

