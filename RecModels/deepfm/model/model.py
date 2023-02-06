import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(
        self,
        channels):
        super().__init__()
        
        layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(channels[-1], 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
        
class DeepFM(nn.Module):
    def __init__(
        self,
        deepfm_params):
        super().__init__()
        
        self.num_cats = deepfm_params.get('num_cats', [])
        self.ds_dim = deepfm_params.get('ds_dim', None)
        self.sp_dim = deepfm_params.get('sp_dim', None)
        self.embed_dim = deepfm_params.get('embed_dim', 8)
        self.hid_channels = deepfm_params.get('hid_channels',[])
        self.l2_reg_ds = deepfm_params.get('l2_reg_ds', 0.0)
        self.l2_reg_sp = deepfm_params.get('l2_reg_sp', 0.0)
        
        # Initialize Embedding for linear part of dense, 1st and 2nd part of sparse_cols
        self.fm_fc = nn.Linear(self.ds_dim, 1)
        
        self.embedders_1st = nn.ModuleList()
        for i in range(self.sp_dim):
            self.embedders_1st.append(nn.Embedding(self.num_cats[i], 1))
            nn.init.xavier_uniform_(self.embedders_1st[i].weight)
            
        self.embedders_2nd = nn.ModuleList()
        for i in range(self.sp_dim):
            self.embedders_2nd.append(nn.Embedding(self.num_cats[i], self.embed_dim))
            nn.init.xavier_uniform_(self.embedders_2nd[i].weight)
        
        # Initialize dnn part
        
        dense_input_dim = self.ds_dim + self.sp_dim * self.embed_dim
        channels = [dense_input_dim] + self.hid_channels
        
        self.dnn = DNN(channels)
        
        self.final_bias = nn.Parameter(torch.randn((1, )))
        
    def forward(self, ds_input, sp_input):
        
        # Calculate fm part
        sp1st = []
        sp2nd = []
        for i in range(self.sp_dim):
            sp1st.append(self.embedders_1st[i](sp_input[:, i: i+1]))
            sp2nd.append(self.embedders_2nd[i](sp_input[:, i: i+1]))
        
        fm_out = self.cal_fm(sp1st, sp2nd, ds_input)
        
        # Calculate dnn part
        
        dnn_sp_in = torch.cat(sp2nd, dim=-1)
        dnn_sp_in = dnn_sp_in.reshape(dnn_sp_in.size(0), -1)
        
        dnn_in = torch.cat([dnn_sp_in, ds_input], dim=-1)
        dnn_out = self.dnn(dnn_in)
        
        return torch.sigmoid(dnn_out + fm_out + self.final_bias)
    
    def cal_fm(self, sp1st, sp2nd, ds_input):
        
        cross_part = torch.cat(sp2nd, dim=1)# bs, field, embed_dim
        cross_part_1 = torch.pow(torch.sum(cross_part, dim=1, keepdim=True), 2)
        cross_part_2 = torch.sum(torch.pow(cross_part, 2), dim=1, keepdim=True)
        
        cross_out = torch.sum(0.5 * (cross_part_1 - cross_part_2), dim=-1, keepdim=False)
        linear_out = torch.sum(torch.cat(sp1st, dim=-1), dim=-1)
        dense_out = self.fm_fc(ds_input)
        
        fm_out = cross_out + linear_out + dense_out
        return fm_out
    
    def cal_weight_reg(self, loss):
        loss += self.l2_reg_ds * torch.norm(self.fm_fc.weight)
        
        for embedder in self.embedders_1st:
            loss += self.l2_reg_sp * torch.norm(embedder.weight)
        for embedder in self.embedders_2nd:
            loss += self.l2_reg_sp * torch.norm(embedder.weight)
            
        return loss
        