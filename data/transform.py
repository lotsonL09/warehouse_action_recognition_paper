import torch
import torchvision.transforms.v2.functional as F
import torchvision.transforms.v2 as v2
from torchvision.transforms import ColorJitter

class VideoTrainTransform:
    def __init__(self, size=96, 
                 brightness=0.2,contrast=0.2,
                 saturation=0.1,hue=0.02):
        """
        hue -> [-0.5,0.5]
        """
        self.size = size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )


    def __call__(self, video):
        video = video.to(torch.float32) / 255.0

        i, j, h, w = v2.RandomResizedCrop.get_params(
            video[0], scale=(0.8, 1.0), ratio=(3/4, 4/3)
        )

        video = F.resized_crop(video, i, j, h, w, size=(self.size, self.size))

        if torch.rand(1) < 0.5:
            video = F.horizontal_flip(video)

        video = torch.stack([self.color_jitter(frame) for frame in video])

        video = F.normalize(video, mean=self.mean, std=self.std)

        return video
    
class VideoValTransform:
    def __init__(self, resize_size=110, crop_size=96):
        self.resize_size = resize_size
        self.crop_size = crop_size

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