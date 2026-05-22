from dataclasses import dataclass
from pydantic import BaseModel

class Selector(BaseModel):
    strategy:str
    candidate_frames:int | None = None
    selected_frames: int 

class Dataset(BaseModel):
    root: str
    selector: Selector
    resize_size:int
    crop_size:int

class Dataloader(BaseModel):
    batch_size:int
    num_workers:int

class DataConfig(BaseModel):
    dataset: Dataset
    dataloader: Dataloader