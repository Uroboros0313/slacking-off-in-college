import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProduct(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(
        self, 
        dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = F.sigmoid(torch.mm(z, z.t()))
        return adj


class GCN(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        dropout):
        super().__init__()
        
        self.fc = nn.Linear(in_c, out_c)
        self.dropout = dropout
        
    def foward(self, x, support):
        x = F.dropout(x, self.dropout, self.training)
        msg = torch.spmm(x, support)
        msg = self.fc(msg)
        return F.relu(msg)