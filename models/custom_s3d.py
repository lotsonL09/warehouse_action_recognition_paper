from torchvision.models.video import s3d,S3D_Weights
import torch.nn as nn

def get_custom_s3d(num_classes:int,phase:int):
    s3d_model = s3d(weights=S3D_Weights.DEFAULT)
    match phase:
        case 1:

            #Freeze weights
            for param in s3d_model.parameters():
                param.requires_grad=False

            num_ftrs = s3d_model.classifier[1].in_channels

            conv3d_classifier=nn.Conv3d(num_ftrs,
                                        num_classes,
                                        kernel_size=(1,1,1),
                                        stride=(1,1,1))
            
            s3d_model.classifier[1]=conv3d_classifier

            for param in s3d_model.classifier.parameters():
                param.requires_grad=True
        
        case 2:

            #Freeze weights
            for param in s3d_model.parameters():
                param.requires_grad=False

            num_ftrs = s3d_model.classifier[1].in_channels

            conv3d_classifier=nn.Conv3d(num_ftrs,
                                        num_classes,
                                        kernel_size=(1,1,1),
                                        stride=(1,1,1))
            
            s3d_model.classifier[1]=conv3d_classifier

            for param in s3d_model.features[14:].parameters():
                param.requires_grad=True
            
            for param in s3d_model.avgpool.parameters():
                param.requires_grad=True

            for param in s3d_model.classifier.parameters():
                param.requires_grad=True

        case 3:

            #Freeze weights
            for param in s3d_model.parameters():
                param.requires_grad=False

            num_ftrs = s3d_model.classifier[1].in_channels

            conv3d_classifier=nn.Conv3d(num_ftrs,
                                        num_classes,
                                        kernel_size=(1,1,1),
                                        stride=(1,1,1))
            
            s3d_model.classifier[1]=conv3d_classifier

            for param in s3d_model.features[8:].parameters():
                param.requires_grad=True
            
            for param in s3d_model.avgpool.parameters():
                param.requires_grad=True

            for param in s3d_model.classifier.parameters():
                param.requires_grad=True

        case 4:
            # unfreeze all weights
            
            num_ftrs = s3d_model.classifier[1].in_channels

            conv3d_classifier=nn.Conv3d(num_ftrs,
                                        num_classes,
                                        kernel_size=(1,1,1),
                                        stride=(1,1,1))
            
            s3d_model.classifier[1]=conv3d_classifier

            for param in s3d_model.parameters():
                param.requires_grad=True

    return s3d_model


    


