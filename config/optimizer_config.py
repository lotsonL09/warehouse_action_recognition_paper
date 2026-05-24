from pydantic import BaseModel

class OptimizerConfig(BaseModel):
    name: str
    lr: float
    weight_decay:float | None = None
    momentum: float | None = None