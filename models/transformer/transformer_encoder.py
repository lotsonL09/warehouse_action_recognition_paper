import torch.nn as nn
import torch

class TransformerEncoder(nn.Module):
    def __init__(self, d_model,
                num_heads,
                num_layers,
                mlp_size:int,
                activaction:str='gelu',
                batch_first:bool=True,
                norm_first:bool=True,
                dropout:int=0.1):
        super().__init__()
        self.transformer_encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                                nhead=num_heads,
                                                                dim_feedforward=mlp_size,
                                                                dropout=dropout,
                                                                activation=activaction,
                                                                batch_first=batch_first,
                                                                norm_first=norm_first)
        
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                    num_layers=num_layers)
    
    def forward(self,x:torch.tensor):
        x=self.transformer_encoder(x)
        return x