from pydantic import BaseModel

class TrainConfig(BaseModel):
    num_classes:int
    epochs: int
    warmup_epochs:int | None