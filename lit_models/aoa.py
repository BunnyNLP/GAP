from turtle import forward
import torch
import torch.nn as nn

import numpy as np

class AOA(nn.Module):
    def __init__(self,d_k) -> None:
        super().__init__()
        self.scale = np.power(d_k,0.5)
        self.row_softmax = nn.Softmax(dim=2)
        self.col_softmax = nn.Softmax(dim=1)

    def forward(self,raw_embedding , label_embedding):
        u = torch.bmm(label_embedding, raw_embedding.transpose(1,2))#torch.Size([16, 40, 256])
        u = u /self.scale
        
        row_attn = self.row_softmax(u)#torch.Size([16, 40, 256])
        col_attn = self.col_softmax(u)#torch.Size([16, 40, 256])

        row_mean = torch.mean(row_attn,dim=1,keepdim=True)#torch.Size([16, 1, 256])
        col_mean = torch.mean(col_attn,dim=2,keepdim=True)#torch.Size([16, 40, 1])

        input_att = torch.bmm(row_attn.transpose(1,2) ,col_mean).squeeze(2)# #torch.Size([16, 256, 40])*  torch.Size([16, 40, 1])  -> torch.Size([16, 256, 1]) ->torch.Size([16, 256])
        # label_att = torch.bmm(col_attn,row_mean.transpose(1,2)).squeeze(2)#torch.Size([16, 40, 256]) * torch.Size([16, 256, 1]) -> torch.Size([16, 40, 1]) ->torch.Size([16, 40 ])
        label_att = col_mean.squeeze(-1)
        return  input_att,label_att

if __name__ =="__main__":
    model = AOA(1024)
    label_embedding = torch.rand(16,40,1024)
    raw_embedding = torch.rand(16,256,1024)
    label_att , input_att = model(raw_embedding,label_embedding)
