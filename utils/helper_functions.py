import subprocess
import random
import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader,cpu
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torch
import json
import shutil


def shape_hook(name):
    def hook(module,input,output):
        print(name)
        print(f"input shape: {input[0].shape}")
        print(f"output shape: {output.shape}")
        print("-"*40)
    return hook

'''
stages=[
    resnet_model.conv1,
    resnet_model.layer1,
    resnet_model.layer2,
    resnet_model.layer3,
    resnet_model.layer4
]
'''

def monitor_model(stages:list,resnet_model:nn.Module,tensor_dim:tuple):
    handles = [
        stage.register_forward_hook(shape_hook(stage.__class__.__name__))
        for stage in stages
    ]

    dummy_tensor=torch.randn(tensor_dim)

    resnet_model(dummy_tensor).shape

    for h in handles:
            h.remove()

def reencode_and_resize_ffmpeg(input_path:Path,
                               output_path:Path,
                               short_side:int=256,
                               crf:int=10,
                               preset:str="fast") -> bool:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scale_filter = (
        f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        "-an",
        str(output_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"Error: {input_path}")

def collect_videos(input_dir:Path):
    videos=list(input_dir.glob("*/*/*.mp4"))
    return videos

def process_dataset(input_dir:Path,
                    output_dir:Path,
                    short_side:int=256):
    videos = collect_videos(input_dir)

    for video in tqdm(videos, desc="Procesando", unit="video"):
        rel = video.relative_to(input_dir)
        
        out_path = (output_dir / rel).with_suffix(".mp4")

        reencode_and_resize_ffmpeg(
            input_path=video,
            output_path=out_path,
            short_side=short_side
        )

def plot_images(frames,row_num,col_num,gray_scale=False):
    num_frames = row_num * col_num

    fig, axes = plt.subplots(row_num, col_num, figsize=(4*col_num, 4*row_num))

    if row_num * col_num == 1:
        axes = [axes]  # convertir en lista
    else:
        axes = axes.flatten()

    for i in range(num_frames):
        if i < len(frames):
            if gray_scale:
                axes[i].imshow(frames[i],cmap="gray")
            else:
                axes[i].imshow(frames[i])
            axes[i].set_title(f"Frame {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def get_samples_from_dataset(dataset_dir:Path):
    frames=[]
    for activity in dataset_dir.iterdir():
        videos=[video for video in activity.glob("*/*.mp4")]
        samples=random.sample(videos,5)

        for video_path in samples:
            vr=VideoReader(str(video_path),ctx=cpu(0))
            frame=vr[45].asnumpy()
            frames.append(frame)

    return np.stack(frames,axis=0)
                
def save_experiment(results:dict,
                    config_path:Path,
                    save_path:Path):
    
    results_path=save_path/"metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    shutil.copy(config_path,save_path/"config.yaml")

def debug_dataloader(dataloader:torch.utils.data.DataLoader):

    print("========== DATALOADER INFO ==========")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Num batches: {len(dataloader)}")
    print(f"Num workers: {dataloader.num_workers}")
    batch = next(iter(dataloader))
    videos, labels = batch

    print("\n========== BATCH INFO ==========")
    print(f"Videos shape: {videos.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Videos dtype: {videos.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    print("================================\n")

def set_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

