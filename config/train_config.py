from pydantic import BaseModel

class TrainConfig(BaseModel):
    num_classes:int
    epochs: int |None = None
    warmup_epochs:int | None = None