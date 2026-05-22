from pydantic import BaseModel

class TransformerEncoder(BaseModel):
    num_heads:int
    num_layers:int
    mlp_size:int

class MLP(BaseModel):
    hidden_units_mlp: list[int]

class Adapter(BaseModel):
    factor:int
    dropout:float

class CNN_class(BaseModel):
    embedding_dim:int


class ModelConfig(BaseModel):
    cnn:CNN_class
    adapter:Adapter
    transformer_encoder:TransformerEncoder
    mlp:MLP
    dropout:float
