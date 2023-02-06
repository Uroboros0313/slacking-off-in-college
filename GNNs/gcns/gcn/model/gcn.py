import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(
        self,
        in_c,
        out_c):
        super().__init__()
        
        self.fc = nn.Linear(in_c, out_c)
        
    def forward(self, x, adj):
        x = self.fc(x)
        return torch.spmm(adj, x)
        
        
class GCN(nn.Module):
    def __init__(
        self,
        in_c,
        hid_c,
        out_c,
        dropout=0.5,
        ):
        super().__init__()
        self.net=nn.ModuleList([
            GraphConv(in_c, hid_c),
            GraphConv(hid_c, out_c)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = F.relu(self.net[0](x, adj))
        x = self.dropout(x)
        x = self.net[1](x, adj)
        
        return torch.log_softmax(x, dim=1)
            