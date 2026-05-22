from config.load_config import load_config
from training.optimizer_scheduler import define_optimizer_and_scheduler
from models.action_model import ActionRecognitionModelWithoutSelector,ActionRecognitionModelWithMotionSelector
from data.video_dataset import create_datasets,create_action_triple_datasets
from data.video_dataloader import load_dataloader
from data.transform import VideoTrainTransform,VideoValTransform
from training.train_and_evaluate import train_and_evaluate,train_and_evaluate_triplets
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

    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg=cfg.data
    model_cfg=cfg.model
    train_cfg=cfg.train

    seeds=cfg.experiment.seeds

    save_path=root/cfg.experiment.save_dir/cfg.experiment.name
    save_path.mkdir(parents=True,exist_ok=True)

    root_path=root/cfg.data.dataset.root

    train_path=root_path/"train"
    val_path=root_path/"val"


    ## AQUI DEBERIAMOS DEFINIR EL BUCLE PARA PROBAR VARIAS SEMILLAS

    for seed in seeds:
        set_seed(seed=seed)

        seed_path=save_path/str(seed)
        seed_path.mkdir(parents=True,exist_ok=True)

        train_dataset,val_dataset=create_action_triple_datasets(train_path=train_path,
                                                             val_path=val_path,
                                                             num_frames=data_cfg.dataset.selector.selected_frames,
                                                             strategy=data_cfg.dataset.selector.strategy, ##IMPORTANTE
                                                             train_transform=VideoTrainTransform(),
                                                            val_transform=VideoValTransform())  


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
        
        _,embd_val_dataset=create_datasets(train_path=train_path,
                        val_path=val_path,
                        num_frames=data_cfg.dataset.selector.selected_frames,
                        strategy=data_cfg.dataset.selector.strategy, ##IMPORTANTE
                        train_transform=VideoTrainTransform(),
                        val_transform=VideoValTransform())  
        
        embd_val_dataloader=load_dataloader(train_dataset=None,
                                            val_dataset=embd_val_dataset,
                                            batch_size=data_cfg.dataloader.batch_size,
                                            is_train_loader=False,
                                            num_workers=data_cfg.dataloader.num_workers,
                                            seed=seed)
        
        model=None

        if data_cfg.dataset.selector.strategy == "motion-distribution":
            if data_cfg.dataset.selector.selected_frames == None:
                raise Exception("Verificar las variables candidate_frames y selected_frames")
            model= ActionRecognitionModelWithMotionSelector(num_classes=train_cfg.num_classes,
                                                    selected_frames=data_cfg.dataset.selector.selected_frames,
                                                    embedding_dim=model_cfg.cnn.embedding_dim,
                                                    num_heads=model_cfg.transformer_encoder.num_heads,
                                                    num_layers=model_cfg.transformer_encoder.num_layers,
                                                    mlp_size=model_cfg.transformer_encoder.mlp_size,
                                                    dropout=model_cfg.dropout,
                                                    adapter_dropout=model_cfg.adapter.dropout,
                                                    adapter_factor=model_cfg.adapter.factor)
        else:
            model = ActionRecognitionModelWithoutSelector(num_classes=train_cfg.num_classes,
                                                        num_frames=data_cfg.dataset.selector.selected_frames,
                                                        embedding_dim=model_cfg.cnn.embedding_dim,
                                                        num_heads=model_cfg.transformer_encoder.num_heads,
                                                        num_layers=model_cfg.transformer_encoder.num_layers,
                                                        mlp_size=model_cfg.transformer_encoder.mlp_size,
                                                        dropout=model_cfg.dropout,
                                                        adapter_dropout=model_cfg.adapter.dropout,
                                                        adapter_factor=model_cfg.adapter.factor)


        optimizer,scheduler=define_optimizer_and_scheduler(model=model,
                                    learning_rate=train_cfg.lr,
                                    weight_decay=train_cfg.weight_decay,
                                    scheduler=train_cfg.scheduler,
                                    warmup_epochs=train_cfg.warmup_epochs,
                                    total_epochs=train_cfg.epochs,
                                    train_datalaoder=train_dataloader)

        loss_fcn=nn.TripletMarginLoss(margin=1.0,p=2)

        scheduler_name="" if train_cfg.scheduler == None else train_cfg.scheduler.name

        results=train_and_evaluate_triplets(model=model,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        emb_val_dataloader=embd_val_dataloader,
                        loss_fcn=loss_fcn,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scheduler_name=scheduler_name,
                        n_epochs=train_cfg.epochs,
                        save_path=seed_path,
                        device=device)

        save_experiment(results=results,
                        config_path=config_path,
                        save_path=seed_path)
        


if __name__ == "__main__":
    main()