import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as F
import torchvision.transforms.v2 as v2
from torchvision.transforms import ColorJitter

'''
* S3D:
    - resize = 256 min side
    - random crops = (224,224)
    - Horizontal flip = 0.5

* VideoResNet: Input size = (16,3,112,112) bs x channels x height x width
    - resize = 128 min side
    - random crops = (112,112)
    - Horizontal flip = 0.5

* MVIT:
    - resize = 256 min side
    - random crops = (224,224)
    - Horizontal flip = 0.5

* VivitForVideoClassification:
    - resize = 256 min side
    - random crops = (224,224)
    - Horizontal flip = 0.5

* VideoSwinTransformer:
    - resize = 256 min side
    - random crops = (224,224)
    - Horizontal flip = 0.5
'''


class VideoTrainTransform:
    def __init__(self,
                 model:nn.Module, 
                 brightness:float=0.4,
                 contrast:float=0.4,
                 saturation:float=0.4,
                 hue:float=0.1):
        """
        hue -> [-0.5,0.5]
        """
        self.model_name=model.__class__.__name__

        match self.model_name:
            case "S3D" | "MViT" | "VivitForVideoClassification" | "SwinTransformer3d":
                self.resize_dim=256 #256
                self.crop_dim=224 #224
            case "VideoResNet":
                self.resize_dim=128
                self.crop_dim=112
            case _:
                raise ValueError(f"Unsupported model: {self.model_name}")

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )


    def __call__(self, video):

        # 1. Normalizar a [0,1]
        video = video.to(torch.float32) / 255.0

        _,_,h,w = video.shape


        # 2. Resize manteniendo la relacion de aspecto

        if h<w:
            new_h = self.resize_dim
            new_w=int(w*self.resize_dim/h)
        else:
            new_w = self.resize_dim
            new_h=int(h*self.resize_dim/w)
        
        video = F.resize(video, [new_h, new_w])


        # 3. RandomCrop

        video = F.center_crop(video, [self.crop_dim, self.crop_dim])

        # 4. RandomHorizontalFlip

        if torch.rand(1) < 0.5:
            video = F.horizontal_flip(video)

        # 5. ColorJitter (aplicar a cada frame)

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            ColorJitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue
            )
        )

        for fn_id in fn_idx:

            if fn_id == 0 and brightness_factor is not None:
                video = F.adjust_brightness(video, brightness_factor)

            elif fn_id == 1 and contrast_factor is not None:
                video = F.adjust_contrast(video, contrast_factor)

            elif fn_id == 2 and saturation_factor is not None:
                video = F.adjust_saturation(video, saturation_factor)

            elif fn_id == 3 and hue_factor is not None:
                video = F.adjust_hue(video, hue_factor)

        # 6. Normalización final

        # video = F.normalize(video, mean=self.mean, std=self.std)

        return video
    
class VideoValTransform:
    def __init__(self, model:nn.Module):
        self.model_name = model.__class__.__name__

        match self.model_name:
            case "S3D" | "MViT" | "VivitForVideoClassification" | "VideoSwinTransformer":
                self.resize_size = 224
                self.crop_size = 224
            case "VideoResNet":
                self.resize_size = 128
                self.crop_size = 112
            case _:
                raise ValueError(f"Unsupported model: {self.model_name}")

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, video):
        """
        video: Tensor (T, C, H, W) uint8
        """

        # Convertir a float y escalar
        video = video.to(torch.float32) / 255.0

        # Resize y CenterCrop (consistente en todos los frames)
        video = F.resize(video, self.resize_size)
        video = F.center_crop(video, [self.crop_size, self.crop_size])

        # Normalización ImageNet
        video = F.normalize(video, mean=self.mean, std=self.std)

        return video