from config.load_config import load_config
from models import get_model
from data.video_dataset import create_datasets_tensor
from data.video_dataloader import load_dataloader
from data.transform import VideoTrainTransformTensor,VideoValTransformTensor
from training.model_evaluation import evaluate_model
from utils.helper_functions import set_seed
from pathlib import Path
import torch
import torch.nn as nn

'''
uniform
random
motion-distribution
'''

def main():

    root=Path.cwd()

    configs_path = Path("configs")

    config_path=configs_path/"base_evaluation.yaml"

    experiment_cfg = load_config(config_path)

    experiment_phase=experiment_cfg.experiment.phase

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    data_cfg=experiment_cfg.data
    train_cfg=experiment_cfg.train

    seeds=experiment_cfg.experiment.seeds

    save_path=root/experiment_cfg.experiment.save_dir/experiment_cfg.experiment.name

    root_path=root/data_cfg.dataset.root

    train_path=root_path/"train"
    val_path=root_path/"val"

    for seed in seeds:
        model = get_model(model=experiment_cfg.experiment.model,
                          num_classes=train_cfg.num_classes,
                          phase=experiment_phase).to(device)
        
        print(f"model: {model.__class__.__name__}")

        print(f"seed: {seed}")

        set_seed(seed=seed)

        seed_path=save_path/str(seed)

        _,val_dataset=create_datasets_tensor(model_name=model.__class__.__name__,
                                                train_path=train_path,
                                                val_path=val_path,
                                                train_transform=VideoTrainTransformTensor(model=model),
                                                val_transform=VideoValTransformTensor(model=model))        

        val_dataloader=val_dataloader=load_dataloader(train_dataset=None,
                        val_dataset=val_dataset,
                        batch_size=data_cfg.dataloader.batch_size,
                        is_train_loader=False,
                        num_workers=data_cfg.dataloader.num_workers,
                        seed=seed)

        print(f"save path: {seed_path}")

        evaluate_model(model=model,
                       model_save_path=seed_path/"best_checkpoint.pth",
                       val_dataloader=val_dataloader,
                       num_classes=train_cfg.num_classes,
                       device=device)

        

if __name__ == "__main__":
    main()