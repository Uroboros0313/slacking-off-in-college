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
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

        
class NFM(nn.Module):
    def __init__(
        self,
        nfm_params):
        super().__init__()
        
        self.num_cats = nfm_params.get('num_cats', [])
        self.ds_dim = nfm_params.get('ds_dim', None)
        self.sp_dim = nfm_params.get('sp_dim', None)
        self.embed_dim = nfm_params.get('embed_dim', 8)
        self.hid_channels = nfm_params.get('hid_channels',[])
        self.bi_dropout = nfm_params.get('bi_dropout',0.2)
        
        lin_embedders = nn.ModuleList()
        for i in range(self.sp_dim):
            lin_embedders.append(nn.Embedding(self.num_cats[i], 1))
        self.lin_embedders = lin_embedders
        
        embedders = nn.ModuleList()
        for i in range(self.sp_dim):
            embedders.append(nn.Embedding(self.num_cats[i], self.embed_dim))
            nn.init.xavier_normal_(embedders[i].weight)
        self.embedders = embedders
        
        self.lin_part = nn.Linear(self.ds_dim, 1)
        
        self.dropout = nn.Dropout(self.bi_dropout)
        dnn_input_dim = self.embed_dim + self.ds_dim
        self.dnn = DNN([dnn_input_dim] + self.hid_channels)
        self.final_linear = nn.Linear(self.hid_channels[-1], 1)
        
    def forward(self, ds_input, sp_input):
        
        sp_inputs = []
        lin_inputs = []
        for i in range(self.sp_dim):
            sp_inputs.append(self.embedders[i](sp_input[:, i: i+1]))
            lin_inputs.append(self.lin_embedders[i](sp_input[:, i]))
        sp_inputs = torch.cat(sp_inputs, dim=1) # batch_size, field, channel
        lin_output = self.lin_part(ds_input) + torch.sum(torch.cat(lin_inputs, dim=1), dim=1, keepdim=True)
        
        bi_pooled = self.bi_interaction_pooling(sp_inputs)
        dnn_input = torch.cat([bi_pooled, ds_input], dim=1)
        dnn_out = self.dnn(dnn_input)
        
        out = torch.sigmoid(self.final_linear(dnn_out) + lin_output)
        return out
    
    def bi_interaction_pooling(self, sp_inputs):
        bi_part1 = torch.pow(torch.sum(sp_inputs, dim=1, keepdim=True), 2)
        bi_part2 = torch.sum(torch.pow(sp_inputs, 2), dim=1, keepdim=True)
        bi_pooled = (0.5 * (bi_part1 - bi_part2)).squeeze(1)
        
        if self.bi_dropout > 0:
            self.dropout(bi_pooled)
        return bi_pooled
    
        
        
    

        