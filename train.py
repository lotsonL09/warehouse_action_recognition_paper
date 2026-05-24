from config.load_config import load_config, load_optim_scheduler_config

from models import get_model

from data.video_dataset import create_datasets
from data.video_dataloader import load_dataloader
from data.transform import VideoTrainTransform,VideoValTransform
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

    config_path=Path("configs/base.yaml")

    experiment_cfg = load_config(config_path)

    optimizer_scheduler_cfg=load_optim_scheduler_config(experiment_cfg.experiment.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    data_cfg=experiment_cfg.data
    train_cfg=experiment_cfg.train

    model = get_model(model=experiment_cfg.experiment.model,num_classes=train_cfg.num_classes).to(device)

    seeds=experiment_cfg.experiment.seeds

    save_path=root/experiment_cfg.experiment.save_dir/experiment_cfg.experiment.name
    save_path.mkdir(parents=True,exist_ok=True)

    root_path=root/data_cfg.dataset.root

    train_path=root_path/"train"
    val_path=root_path/"val"


    ## AQUI DEBERIAMOS DEFINIR EL BUCLE PARA PROBAR VARIAS SEMILLAS

    for seed in seeds:
        set_seed(seed=seed)

        seed_path=save_path/str(seed)
        seed_path.mkdir(parents=True,exist_ok=True)

        train_dataset,val_dataset=create_datasets(model_name=model.__class__.__name__,
                                                train_path=train_path,
                                                val_path=val_path,
                                                num_frames=data_cfg.dataset.selector.selected_frames,
                                                strategy=data_cfg.dataset.selector.strategy, ##IMPORTANTE
                                                train_transform=VideoTrainTransform(model=model),
                                                val_transform=VideoValTransform(model=model))

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

        loss_fcn=nn.CrossEntropyLoss() #TODO: LABEL SMOOTHING

        results=train_and_evaluate(model=model,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        loss_fcn=loss_fcn,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        n_epochs=train_cfg.epochs,
                        save_path=seed_path,
                        device=device)

        save_experiment(results=results,
                        config_path=config_path,
                        save_path=seed_path)
        


if __name__ == "__main__":
    main()