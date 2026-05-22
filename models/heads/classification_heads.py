import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self,embedding_dim:int=1024,hidden_units:list=[256,128],dropout:float=0.1,num_classes:int=3):
        super().__init__()
        self.linear_1=nn.Linear(in_features=embedding_dim,out_features=hidden_units[0])
        self.linear_2=nn.Linear(in_features=hidden_units[0],out_features=hidden_units[1])
        self.linear_3=nn.Linear(in_features=hidden_units[1],out_features=num_classes)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor):
        x=self.dropout(self.relu(self.linear_1(x)))

        x=self.dropout(self.relu(self.linear_2(x)))

        return self.linear_3(x)