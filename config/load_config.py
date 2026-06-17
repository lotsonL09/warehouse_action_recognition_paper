from .data_config import DataConfig
from .train_config import TrainConfig
from .experiment_config import ExperimentConfig
from .optimizer_config import OptimizerConfig
from .scheduler_config import SchedulerConfig

import yaml
from pathlib import Path
from pydantic import BaseModel

class Config(BaseModel):
    experiment: ExperimentConfig
    data: DataConfig
    train: TrainConfig

class OptimSchedulerConfig(BaseModel):
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

def load_file(path:Path) -> dict:
    with open(path,'r') as f:
        return yaml.safe_load(f)

def load_config(path:Path) -> Config:
    cfg_dict = load_file(path)

    return Config(
        experiment=ExperimentConfig(**cfg_dict['experiment']),
        train=TrainConfig(**cfg_dict['train']),
        data=DataConfig(**cfg_dict['data'])
    )


'''
Model
1: MViT
2: S3D
3: Video ResNet
4: Video Swin Transformer
5: ViViT
'''

model_configs={
    1: "mvit_config.yaml",
    2: "s3d_config.yaml",
    3: "video_resnet_config.yaml",
    4: "video_swin_config.yaml",
    5: "vivit_config.yaml"
}

def load_optim_scheduler_config(model:int):
    config_dir=Path("configs")
    
    model_cfg_dir=config_dir / model_configs[model]
        
    cfg=load_file(model_cfg_dir)

    return OptimSchedulerConfig(
        optimizer=OptimizerConfig(**cfg['optimizer']),
        scheduler=SchedulerConfig(**cfg['scheduler'])
    )


# if __name__ == "__main__":
#     cfg = load_config("configs/base.yaml")
#     print(f"Data config: {cfg.data}")
#     print(f"Experiment config: {cfg.experiment}")
#     print(f"Model: {cfg.experiment.model}")
#     optim_scheduler_cfg=load_optim_scheduler_config(cfg.experiment.model)
#     print(f"Optimizer config: {optim_scheduler_cfg.optimizer}")
#     print(f"Scheduler config: {optim_scheduler_cfg.scheduler}")

