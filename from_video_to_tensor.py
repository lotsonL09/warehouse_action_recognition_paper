from pathlib import Path

from decord import VideoReader,cpu

from pathlib import Path

from models import get_model

import torch.nn as nn

import torch

import torchvision.transforms.v2.functional as F

import numpy as np

print("hello")

root=Path.cwd()

dataset_dir=root/"datasets"/"trimmed_videos"

destiny_dir=root/"datasets"/"r3d_dataset_tensor"

def sample_frames(total_frames):
        
    N=16 
        
    return np.linspace(0, total_frames - 1, N).astype(int)


class VideoTrainTransform:
    def __init__(self,
                 model:nn.Module):
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

        return video

swin_model = get_model(model=3,num_classes=5,pre_trained=True)

transform = VideoTrainTransform(model=swin_model)

def convert_videos_to_numpy_frames(target_dir:Path, destiny_dir:Path):

    destiny_dir.mkdir(exist_ok=True)

    videos = list(target_dir.glob("*/*/*.mp4"))

    for video in videos:
        
        parents=video.parents

        activity=parents[0].name

        person=parents[1].name

        vr=VideoReader(str(video),cpu(0))
        
        total_frames = len(vr)
        
        indices = sample_frames(total_frames)

        frames = vr.get_batch(indices).asnumpy()

        frames = torch.from_numpy(frames)

        frames=frames.permute(0,3,1,2)

        frames=transform(frames)

        frames.permute(0,2,3,1)

        activity_dir=destiny_dir/person/activity
        
        activity_dir.mkdir(parents=True,exist_ok=True)

        video_name=f"{video.name.split('.')[0]}.pt"

        destiny_file=activity_dir/video_name

        frames=(frames*255).to(torch.uint8)

        torch.save(frames,destiny_file)

if __name__ == "__main__":
    convert_videos_to_numpy_frames(dataset_dir,destiny_dir)

