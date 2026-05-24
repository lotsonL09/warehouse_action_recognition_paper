from pydantic import BaseModel

class LinearLr(BaseModel):
    start_factor:float
    end_factor:float

class CosineAnnealingLr(BaseModel):
    eta_min:float

class StepLR(BaseModel):
    factor: int
    gamma: float

class ReduceLrOnPlateau(BaseModel):
    mode: str
    factor: float
    patience: int

class SchedulerConfig(BaseModel):
    name: str
    linear_lr: LinearLr | None = None
    cosine_annealing_lr: CosineAnnealingLr | None = None
    step_lr: StepLR | None = None
    reduce_lr_on_plateau: ReduceLrOnPlateau | None = None