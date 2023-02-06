import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphAttentionLayerV1, GraphAttentionLayerV2


class GATv1(nn.Module):
    def __init__(
        self, 
        in_c, 
        hid_c, 
        out_c, 
        dropout, 
        alpha, 
        nheads):
        super().__init__()
        
        self.dropout = dropout

        self.gat_nheads = nn.ModuleList([GraphAttentionLayerV1(in_c, hid_c, alpha, dropout, ) for _ in range(nheads)])
        self.gat_out = GraphAttentionLayerV1(hid_c * nheads, out_c, dropout=dropout, alpha=alpha)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        gat_nheads_out = []
        for gat_head in self.gat_nheads:
            gat_nheads_out.append(gat_head(x, adj))
            
        h = torch.cat(gat_nheads_out, dim=1)
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.gat_out(F.elu(h), adj)
        
        return torch.log_softmax(out, dim=1)
        

class GATv2(nn.Module):
    def __init__(
        self, 
        in_c, 
        hid_c, 
        out_c, 
        dropout, 
        alpha, 
        nheads):
        super().__init__()
        
        self.dropout = dropout

        self.gat_nheads = nn.ModuleList([GraphAttentionLayerV2(in_c, hid_c, alpha, dropout, ) for _ in range(nheads)])
        self.gat_out = GraphAttentionLayerV1(hid_c * nheads, out_c, dropout=dropout, alpha=alpha)
        
    def forward(self, x, adj):
        h = F.dropout(x, self.dropout, training=self.training)
        gat_nheads_out = []
        for gat_head in self.gat_nheads:
            gat_nheads_out.append(gat_head(x, adj))
            
        gat_nheads_out = torch.cat(gat_nheads_out, dim=1)
        out = self.gat_out(F.elu(gat_nheads_out), adj)
        
        return torch.log_softmax(out, dim=1)