from .resnet_backbone import custom_resnet
from models.transformer.transformer_encoder import TransformerEncoder
from models.transformer.positional_encoding import PositionalEncoding
from models.heads.classification_heads import ClassificationHead
from models.adaptive_selector import filtering_stage
from scipy.signal import savgol_filter
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, input_dim:int,output_dim:int,dropout:float):
        super().__init__()

        self.linear_1=nn.Linear(input_dim,output_dim)
        self.linear_2=nn.Linear(output_dim,output_dim)

        self.batch_norm=nn.BatchNorm1d(output_dim)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor):
        x=self.relu(self.batch_norm(self.linear_1(x)))
        x=self.dropout(x)
        return self.linear_2(x)

class ActionRecognitionModelWithoutSelector(nn.Module):
    def __init__(self,
                num_classes,
                num_frames,
                embedding_dim,
                num_heads,
                num_layers,
                mlp_size,
                dropout,
                adapter_dropout,
                adapter_factor,
                hidden_units_mlp=[256,128]):
        
        super().__init__()
        
        decoder_dim=embedding_dim//adapter_factor

        self.resnet=custom_resnet(embedding_dim=embedding_dim)

        self.pe=PositionalEncoding(d_model=decoder_dim,max_len=num_frames+1)
        
        self.cls_token=nn.Parameter(torch.randn(1,1,decoder_dim))

        self.adapter=Adapter(input_dim=embedding_dim,
                            output_dim=decoder_dim,
                            dropout=adapter_dropout)
        
        self.transformer_encoder=TransformerEncoder(d_model=decoder_dim,
                                                    num_heads=num_heads,
                                                    num_layers=num_layers,
                                                    mlp_size=mlp_size,
                                                    dropout=dropout)
        
        self.mlp = ClassificationHead(embedding_dim=decoder_dim,
                       hidden_units=hidden_units_mlp,
                       num_classes=num_classes)

    def forward(self,x:torch.Tensor):

        bs,n_frames,C,H,W=x.shape
        
        x=x.view(bs*n_frames,C,H,W)
        
        x=self.resnet(x)

        x=self.adapter(x)
        
        x=x.view(bs,n_frames,-1)
        
        cls_token=self.cls_token.expand(bs,-1,-1)
        
        x=torch.cat([cls_token,x],dim=1)
        
        x=x+self.pe(x)
        
        x=self.transformer_encoder(x)
        
        cls_output=x[:,0]
        
        #return self.mlp(cls_output)
        return nn.functional.normalize(cls_output,p=2,dim=1)

class ActionRecognitionModelWithMotionSelector(nn.Module):
    def __init__(self,
                num_classes:int,
                selected_frames:int,
                embedding_dim:int,
                num_heads:int,
                num_layers:int,
                mlp_size:int,
                dropout:int,
                adapter_dropout:float,
                adapter_factor:int,
                hidden_units_mlp:list[int]=[256,128]):
        super().__init__()

        decoder_dim=embedding_dim//adapter_factor

        self.selected_frames=selected_frames

        self.selected_frames=selected_frames
        
        self.resnet=custom_resnet(embedding_dim=embedding_dim)

        self.pe=PositionalEncoding(d_model=decoder_dim,max_len=selected_frames+1)
        
        self.cls_token=nn.Parameter(torch.randn(1,1,decoder_dim))

        self.adapter=Adapter(input_dim=embedding_dim,
                             output_dim=decoder_dim,
                             dropout=adapter_dropout)
        
        self.transformer_encoder=TransformerEncoder(d_model=decoder_dim,
                                                    num_heads=num_heads,
                                                    num_layers=num_layers,
                                                    mlp_size=mlp_size,
                                                    dropout=dropout)
        
        self.mlp = ClassificationHead(embedding_dim=decoder_dim,
                       hidden_units=hidden_units_mlp,
                       num_classes=num_classes)

    def forward(self,x:torch.Tensor):

        bs,n_frames,C,H,W=x.shape

        indices,_=filtering_stage(video_batch=x,n_frames_selected=self.selected_frames)

        print(f"Indices of frames: {indices}")

        indices=indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,C,H,W)

        x=torch.gather(x,1,indices)

        print(f"Shape after filtering frames: {x.shape}")

        x=x.view(bs*self.selected_frames,C,H,W)

        x=self.resnet(x)

        x=self.adapter(x)
        
        x=x.view(bs,self.selected_frames,-1)
        
        cls_token=self.cls_token.expand(bs,-1,-1)
        
        x=torch.cat([cls_token,x],dim=1)
        
        x=x+self.pe(x)
        
        x=self.transformer_encoder(x)
        
        cls_output=x[:,0]
        
        #return self.mlp(cls_output)
        return cls_output