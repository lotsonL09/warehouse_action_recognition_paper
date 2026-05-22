from pydantic import BaseModel

class ReduceLROnPlateauScheduler(BaseModel):
    mode:str
    factor:float
    patience:int

class StepLRScheduler(BaseModel):
    step_size:int
    gamma:float

class CosineAnnealingLR(BaseModel):
    t_max:int
    eta_min:float

class SchedulerClass(BaseModel):
    name:str
    reduceLrOnPlateau:ReduceLROnPlateauScheduler | None
    stepLr:StepLRScheduler | None
    cosineAnnealingLr:CosineAnnealingLR | None


class TrainConfig(BaseModel):
    num_classes:int
    epochs: int
    warmup_epochs:int | None
    lr: float
    weight_decay: float
    optimizer: str
    scheduler: SchedulerClass | None