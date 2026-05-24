# import lightning.pytorch as pl
# import torch.nn as  nn
# import torch.optim as optim
# from torchmetrics.classification import Accuracy
# from models.action_model import ActionRecognitionModelWithoutSelector


# def define_optimizer_and_scheduler(model:nn.Module,learning_rate,weight_decay):
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=learning_rate,
#         weight_decay=weight_decay
#     )

#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.1,
#         patience=2
#     )
#     return optimizer,scheduler    


# class ActionModelWithoutSelectorModule(pl.LightningModule):
#     def __init__(self, num_classes, num_frames,dim_encoder,
#                 num_heads,num_layers,
#                 mlp_size,dropout,
#                 learning_rate=1e-3,weight_decay=1e-2):
#         super().__init__()
#         self.save_hyperparameters()
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.accuracy =  Accuracy(task='multiclass',num_classes=num_classes)
#         self.model = ActionRecognitionModelWithoutSelector(num_classes=num_classes, num_frames=num_frames,
#                                                            dim_encoder=dim_encoder,num_heads=num_heads,
#                                                            num_layers=num_layers,mlp_size=mlp_size,
#                                                            dropout=dropout)
    
#     def forward(self,x):
#         return self.model(x)
    
#     def training_step(self,batch,batch_idx=None):
#         x,y=batch
#         logits=self(x)
#         loss=self.loss_fn(logits,y)
#         return loss
    
#     def validation_step(self, batch,batch_idx=None):
#         x,y=batch
#         logits=self(x)
#         loss=self.loss_fn(logits,y)
#         acc=self.accuracy(logits,y)
#         self.log_dict({'val_loss':loss,'val_acc':acc},prog_bar=True)
    
#     def configure_optimizers(self):
#         optimizer,scheduler=define_optimizer_and_scheduler(
#             self.model,
#             self.hparams.learning_rate,
#             self.hparams.weight_decay
#         )
#         return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'monitor':'val_loss'}}

