from config.load_config import load_config, load_optim_scheduler_config

from models import get_model

from data.video_dataset import create_datasets_tensor
from data.video_dataloader import load_dataloader
from data.transform import VideoTrainTransformTensor,VideoValTransformTensor
from training.train_and_evaluate import train_and_evaluate
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from utils.helper_functions import save_experiment,debug_dataloader,set_seed
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

    config_path=configs_path/"base.yaml"

    experiment_cfg = load_config(config_path)

    optimizer_scheduler_cfg=load_optim_scheduler_config(experiment_cfg.experiment.model)

    experiment_phase=experiment_cfg.experiment.phase

    match experiment_phase:
        case 1:
            print("Linear Probing")
        case 2:
            print("Partial Fine-Tuning I")
        case 3:
            print("Partial Fine-Tuning II")
        case 4:
            print("Full Fine-Tuning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    data_cfg=experiment_cfg.data
    train_cfg=experiment_cfg.train

    seeds=experiment_cfg.experiment.seeds

    save_path=root/experiment_cfg.experiment.save_dir/experiment_cfg.experiment.name
    save_path.mkdir(parents=True,exist_ok=True)

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
        seed_path.mkdir(parents=True,exist_ok=True)

        train_dataset,val_dataset=create_datasets_tensor(model_name=model.__class__.__name__,
                                                train_path=train_path,
                                                val_path=val_path,
                                                train_transform=VideoTrainTransformTensor(model=model),
                                                val_transform=VideoValTransformTensor(model=model))

        train_dataloader=load_dataloader(train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        batch_size=data_cfg.dataloader.batch_size,
                        is_train_loader=True,
                        num_workers=data_cfg.dataloader.num_workers,
                        seed=seed)

        val_dataloader=load_dataloader(train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        batch_size=data_cfg.dataloader.batch_size,
                        is_train_loader=False,
                        num_workers=data_cfg.dataloader.num_workers,
                        seed=seed)
        
        optimizer=build_optimizer(model=model,
                                optimizer_config=optimizer_scheduler_cfg.optimizer)


        scheduler=build_scheduler(model=model,
                                  optimizer=optimizer,
                                  scheduler_config=optimizer_scheduler_cfg.scheduler,
                                  total_epochs=train_cfg.epochs,
                                  warmup_epochs=train_cfg.warmup_epochs
                                  )

        loss_fcn=nn.CrossEntropyLoss()

        results=train_and_evaluate(model=model,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        loss_fcn=loss_fcn,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        n_epochs=train_cfg.epochs,
                        save_path=seed_path,
                        device=device)
        
        
        save_experiment(model=experiment_cfg.experiment.model,
                        results=results,
                        configs_path=configs_path,
                        save_path=seed_path)
        


if __name__ == "__main__":
    main()