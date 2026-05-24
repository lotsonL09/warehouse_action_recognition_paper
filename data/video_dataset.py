from pathlib import Path
from torch.utils.data import Dataset
from decord import VideoReader,cpu
import numpy as np
from .transform import VideoTrainTransform,VideoValTransform
import torch
import random

def find_classes(directory:Path) -> tuple[list[str], dict[str,int]]:

    classes_names = sorted([ entry.name for entry in directory.iterdir() if entry.is_dir()])

    if not classes_names:
        raise FileNotFoundError(f"Couldn't find any classes in {directory} ... please check file structure.")

    class_to_idx={}
    idx_to_class={}

    for i, class_name in enumerate(classes_names):
        class_to_idx[class_name]=i
        idx_to_class[i]=class_name


    return idx_to_class,class_to_idx

'''
uniform
random
motion-distribution
two_stages
'''

class VideoDataset(Dataset):
    def __init__(self,
                model_name:str,
                targ_dir:Path,
                num_frames=10,
                strategy="motion-distribution",
                transform=None):
        
        super().__init__()

        self.model_name=model_name
        
        self.paths=list(targ_dir.glob("*/*.mp4"))

        self.idx_to_class,self.class_to_idx=find_classes(targ_dir)

        self.labels=[ self.class_to_idx[path.parent.name] for path in self.paths]

        self.transform = transform

        self.strategy = strategy

        self.num_frames = num_frames

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):

        path = self.paths[index]

        label = self.labels[index]

        video=VideoReader(str(path),ctx=cpu(0))
        total_frames = len(video)

        if self.strategy == "motion-distribution":
            indices = range(len(video))
        else:
            indices = self.sample_frames(total_frames)

        video=video.get_batch(indices).asnumpy() # frames,H,W,C

        video = torch.from_numpy(video)  # uint8

        video=video.permute(0,3,1,2) #frames,C,H,W

        if self.transform:
            video=self.transform(video)

        if self.model_name != "VivitForVideoClassification":
            video=video.permute(1,0,2,3)

        return video,label
    
    def sample_frames(self,total_frames):
        
        N=self.num_frames

        if self.strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, N).astype(int)

        elif self.strategy == "random":
            if total_frames >= N:
                indices = np.random.choice(total_frames, N, replace=False)
                indices = np.sort(indices)
            else:
                indices = np.linspace(0, total_frames - 1, N).astype(int)
        else:
            raise ValueError("Unknown sampling strategy")

        return indices

def create_datasets(model_name,train_path,val_path,num_frames,strategy,train_transform,val_transform):

    train_dataset=VideoDataset(model_name=model_name,
                                targ_dir=train_path,
                                num_frames=num_frames,
                                strategy=strategy,
                                transform=train_transform)
    
    val_dataset = VideoDataset(model_name=model_name,
                                targ_dir=val_path,
                                num_frames=num_frames,
                                strategy=strategy,
                                transform=val_transform)
    return train_dataset,val_dataset


class ActionTripletDataset(Dataset):
    def __init__(self, targ_dir:Path,
                num_frames=10,
                strategy="motion-distribution",
                transform=None):
        
        self.paths=list(targ_dir.glob("*/*.mp4"))

        self.idx_to_class,self.class_to_idx=find_classes(targ_dir)

        self.labels=[ self.class_to_idx[path.parent.name] for path in self.paths]

        self.transform = transform

        self.strategy = strategy

        self.num_frames = num_frames

        self.class_to_indices={}
        for idx, label in enumerate(self.labels):
            self.class_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # 1. ANCHOR: El video en el índice actual
        anchor_video_path = self.paths[index]
        anchor_label = self.labels[index]

        # 2. POSITIVE: Otro video de la MISMA clase
        positive_index = index
        while positive_index == index: # Evitar elegir el mismo video
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_video_path = self.paths[positive_index]

        # 3. NEGATIVE: Un video de una clase DIFERENTE
        negative_label = random.choice([l for l in self.class_to_indices.keys() if l != anchor_label])
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_video_path = self.paths[negative_index]

        anchor=self.load_video(anchor_video_path)
        positive=self.load_video(positive_video_path)
        negative=self.load_video(negative_video_path)
        
        return anchor, positive, negative, anchor_label
    
    def load_video(self,path:Path):
        vr=VideoReader(str(path),ctx=cpu(0))

        total_frames=len(vr)

        indices=self.sample_frames(total_frames)

        video=vr.get_batch(indices).asnumpy()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        if self.transform:
            video = self.transform(video)

        return video

    def sample_frames(self,total_frames):
        
        N=self.num_frames

        if self.strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, N).astype(int)

        elif self.strategy == "random":
            if total_frames >= N:
                indices = np.random.choice(total_frames, N, replace=False)
                indices = np.sort(indices)
            else:
                indices = np.linspace(0, total_frames - 1, N).astype(int)
        else:
            raise ValueError("Unknown sampling strategy")

        return indices
    

def create_action_triple_datasets(train_path,val_path,num_frames,strategy,train_transform,val_transform):

    train_dataset=ActionTripletDataset(targ_dir=train_path,
                               num_frames=num_frames,
                               strategy=strategy,
                               transform=train_transform)
    
    val_dataset = ActionTripletDataset(targ_dir=val_path,
                               num_frames=num_frames,
                               strategy=strategy,
                               transform=val_transform)
    return train_dataset,val_dataset