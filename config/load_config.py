from dataclasses import dataclass
from .data_config import DataConfig
from .model_without_selector_config import ModelConfig
from .train_config import TrainConfig
from .experiment_config import ExperimentConfig
import yaml
from pathlib import Path

@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

def load_config(path:Path) -> Config:
    with open(path,'r') as f:
        cfg_dict = yaml.safe_load(f)

    return Config(
        experiment=ExperimentConfig(**cfg_dict['experiment']),
        model= ModelConfig(**cfg_dict['model']),
        train=TrainConfig(**cfg_dict['train']),
        data=DataConfig(**cfg_dict['data'])
    )

# if __name__ == "__main__":
#     cfg = load_config("configs/base.yaml")
#     print(f"Data config: {cfg.data}")
#     print(f"Experiment config: {cfg.experiment}")

