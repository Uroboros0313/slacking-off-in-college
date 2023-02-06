from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


        
class AFM(nn.Module):
    def __init__(
        self,
        afm_params):
        super().__init__()
        
        self.num_cats = afm_params.get('num_cats', [])
        self.ds_dim = afm_params.get('ds_dim', None)
        self.sp_dim = afm_params.get('sp_dim', None)
        self.embed_dim = afm_params.get('embed_dim', 8)
        self.dropout = afm_params.get('dropout', 0.2)
        
        lin_embedders = nn.ModuleList()
        for i in range(self.sp_dim):
            lin_embedders.append(nn.Embedding(self.num_cats[i], 1))
        self.lin_embedders = lin_embedders
        self.lin_part = nn.Linear(self.ds_dim, 1)
        
        embedders = nn.ModuleList()
        for i in range(self.sp_dim):
            embedders.append(nn.Embedding(self.num_cats[i], self.embed_dim))
            nn.init.xavier_normal_(embedders[i].weight)
        self.embedders = embedders
        
        self.att_lin = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, 1)
        self.att_dropout = nn.Dropout(self.dropout)
        
        self.final_lin = nn.Linear(self.embed_dim, 1)
        
    def forward(self, ds_input, sp_input):
        
        sp_inputs = []
        lin_inputs = []
        for i in range(self.sp_dim):
            sp_inputs.append(self.embedders[i](sp_input[:, i: i+1]))
            lin_inputs.append(self.lin_embedders[i](sp_input[:, i]))
        
        lin_output = self.lin_part(ds_input) + torch.sum(torch.cat(lin_inputs, dim=1), dim=1, keepdim=True)
        
        sp_inputs = torch.cat(sp_inputs, dim=1) # batch_size, field, channel
        
        rows, cols = [], []
        for r, c in product(range(self.sp_dim), range(self.sp_dim)):
            rows.append(r)
            cols.append(c)
        
        row_embs = sp_inputs[:, rows, :]
        col_embs = sp_inputs[:, cols, :]
        inner_product = row_embs * col_embs
        
        att_scores = F.relu(self.att_lin(inner_product))
        att_scores = torch.softmax(self.proj(att_scores), dim=1)
        
        att_output = torch.sum(att_scores * inner_product, dim=1)
        att_output = self.att_dropout(att_output)
        
        out = torch.sigmoid(self.final_lin(att_output) + lin_output)
        
        return out
        
        
    
        
        
    

        