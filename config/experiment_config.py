from pydantic import BaseModel

class ExperimentConfig(BaseModel):
    name: str
    save_dir:str
    model: int
    seeds:list[int]