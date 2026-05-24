from transformers import VivitForVideoClassification, VivitConfig

import torch.nn as nn

def get_custom_vivit(num_classes):

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

    return vivit_model 

