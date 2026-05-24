# import lightning.pytorch as pl
# from pathlib import Path
# from transform import VideoTrainTransform,VideoValTransform
# from video_dataset import create_datasets
# from video_dataloader import load_dataloader

# class ActionRecognitionDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir:Path,strategy:str,num_frames=10,batch_size=64,train_workers=2,val_workers=2):
#         super().__init__()
#         self.save_hyperparameters()
#         self.data_dir=data_dir
#         self.batch_size=batch_size
#         self.train_transform=VideoTrainTransform()
#         self.val_transform=VideoValTransform()
#         self.num_frames=num_frames
#         self.strategy=strategy
#         self.train_dataset=None
#         self.val_dataset=None

#     def setup(self,stage=None):
#         train_path=self.data_dir/"train"
#         val_path=self.data_dir/"test" #TODO: LUEGO LO CAMBIAMOS

#         self.train_dataset,self.val_dataset = create_datasets(
#             train_path=train_path,
#             val_path=val_path,
#             num_frames=self.num_frames,
#             strategy=self.strategy,
#             train_transform=self.train_transform,
#             val_transform=self.val_transform
#         )

#     def train_dataloader(self):
#         return load_dataloader(train_dataset=self.train_dataset,
#                                val_dataset=self.val_dataset,
#                                batch_size=self.batch_size,
#                                is_train_loader=True,
#                                num_workers=self.hparams.train_workers)
    
#     def val_dataloader(self):
#         return load_dataloader(train_dataset=self.train_dataset,
#                                val_dataset=self.val_dataset,
#                                batch_size=self.batch_size,
#                                is_train_loader=False,
#                                num_workers=self.hparams.val_workers)


