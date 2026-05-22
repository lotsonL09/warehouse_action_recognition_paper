from torch.utils.data import DataLoader
import numpy as np
import torch
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_dataloader(train_dataset,val_dataset,
                    batch_size,is_train_loader,
                    num_workers,seed):
    
    g=torch.Generator()
    g.manual_seed(seed)

    if is_train_loader:
        loader=DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        loader=DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker
        )
    return loader