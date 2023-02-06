import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayerV1(nn.Module):
    def __init__(
        self,
        in_c,
        hid_c,
        alpha,
        dropout,):
        super().__init__()
        
        self.in_c = in_c
        self.hid_c = hid_c
        self.dropout = dropout
        
        self.W = nn.Linear(in_c, hid_c)
        self.a = nn.Parameter(torch.empty(2 * hid_c, 1), )
        nn.init.xavier_uniform_(self.a.data, gain=nn.init.calculate_gain('leaky_relu', alpha))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        
    def forward(self, h, adj):
        Wh = self.W(h)
        att_mx = self._cal_attentions(Wh, adj)
        h_gat = torch.mm(att_mx, Wh)
        
        return h_gat
            
    def _cal_attentions(self, Wh, adj):
        aWh1 = torch.mm(Wh, self.a[ :self.hid_c])
        aWh2 = torch.mm(Wh, self.a[self.hid_c :])
        e = self.leaky_relu(aWh1 + aWh2.T)
        
        zero_vec = -9e15 * torch.ones_like(adj)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        return attention
        
class GraphAttentionLayerV2(nn.Module):
    '''
    Refer to: How Attentive are Graph Attention Networks, ICLR, 2022
    Link: https://arxiv.org/pdf/2105.14491.pdf
    '''
    def __init__(
        self,
        in_c,
        hid_c,
        alpha,
        dropout):
        super().__init__()
        
        self.in_c = in_c
        self.hid_c = hid_c
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.empty(2 * in_c, hid_c), )
        self.a = nn.Linear(hid_c, 1)
        self.fc = nn.Linear(in_c, hid_c)
        nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain('leaky_relu', alpha))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        
    def forward(self, h, adj):
        att_mx = self.cal_attentions(h, adj)
        h_gat = torch.mm(att_mx, self.fc(h))
        
        return h_gat
            
    def cal_attentions(self, h, adj):
        Wh1 = torch.mm(h, self.W[ :self.in_c])
        Wh2 = torch.mm(h, self.W[self.in_c :])
        Wh = Wh1.reshape(*Wh1.shape, 1) + Wh2.T
        Wh = Wh.transpose(-1, -2)
        
        e = self.leaky_relu(self.a(Wh).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(adj)
        e = torch.where(adj > 0, e, zero_vec)
        att_mx = torch.softmax(e, dim=1)
        att_mx = F.dropout(att_mx)
        
        return att_mx
        
        