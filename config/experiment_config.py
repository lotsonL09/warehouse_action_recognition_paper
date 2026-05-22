from pydantic import BaseModel

class ExperimentConfig(BaseModel):
    name: str
    save_dir:str
    seeds:list[int]