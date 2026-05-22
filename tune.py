import optuna
import lightning.pytorch as pl
from optuna.integration import PyTorchLightningPruningCallback
from data.data_module import ActionRecognitionDataModule
from training.lightning_model_module import ActionModelWithoutSelectorModule
from config.load_config import load_config

def objective(trial):
    cfg=load_config("configs/base.yaml")
    dim_encoder=trial.suggest_categorical("dim_encoder",[256,512,1024])
    num_heads=trial.suggest_categorical("num_heads",[2,4,8])
    num_layers=trial.suggest_categorical("num_layers",[4,6,8])
    mlp_size=trial.suggest_categorical("mlp_size",[1024,2048,3072])

    cfg.model_without_selector.transformer_encoder.dim_encoder=dim_encoder
    cfg.model_without_selector.transformer_encoder.num_heads=num_heads
    cfg.model_without_selector.transformer_encoder.num_layers=num_layers
    cfg.model_without_selector.transformer_encoder.mlp_size=mlp_size

    data=cfg.data
    model=cfg.model_without_selector
    train=cfg.train

    model = ActionModelWithoutSelectorModule(num_classes=train.num_classes,
                                             learning_rate=train.lr,
                                             weight_decay=train.weight_decay,
                                             num_frames=data.dataset.num_frames,
                                             dim_encoder=model.transformer_encoder.dim_encoder,
                                             num_heads=model.transformer_encoder.num_heads,
                                             num_layers=model.transformer_encoder.num_layers,
                                             mlp_size=model.transformer_encoder.mlp_size,
                                             dropout=model.dropout)
    
    datamodule=ActionRecognitionDataModule(data_dir=data.dataset.root,
                                           strategy=data.dataset.strategy,
                                           num_frames=data.dataset.num_frames,
                                           batch_size=data.dataloader.batch_size,
                                           train_workers=data.dataloader.num_workers,
                                           val_workers=data.dataloader.num_workers)
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[
            PyTorchLightningPruningCallback(trial,monitor='val_loss')
        ],
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model,datamodule=datamodule)

    return trainer.callback_metrics['val_acc'].item()


if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial.params)