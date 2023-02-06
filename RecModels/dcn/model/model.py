import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(
        self,
        channels,
        dropout,):
        super().__init__()
        
        layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    

class CrossNetVector(nn.Module):
    def __init__(
        self,
        in_c,
        n_cross,
        ):
        super().__init__()
        
        self.n_cross = n_cross
        self.linears = nn.ModuleList()
        self.bias = nn.ParameterList()
        for i in range(n_cross):
            self.linears.append(nn.Linear(in_c, 1, bias=False))
            self.bias.append(nn.Parameter(torch.zeros((in_c))))
            
    def forward(self, x):
        out = x
        for i in range(self.n_cross):
            out = out * self.linears[i](out) + self.bias[i] + out
            
        return out
        
        
class CrossNetMatrix(nn.Module):
    def __init__(
        self,
        in_c,
        n_cross,
        ):
        super().__init__()
        
        self.n_cross = n_cross
        self.linears = nn.ModuleList()
        self.bias = nn.ParameterList()
        for i in range(n_cross):
            self.linears.append(nn.Linear(in_c, in_c, bias=False))
            self.bias.append(nn.Parameter(torch.zeros((in_c))))

    def forward(self, x):
        out = x
        for i in range(self.n_cross):
            out = out * self.linears[i](out) + self.bias[i] + out
            
        return out


class CrossNetMix(nn.Module):
    def __init__(
        self,
        in_c,
        n_cross,
        rank,
        n_experts):
        super().__init__()
        
        self.n_experts = n_experts
        self.rank = rank
        self.in_c = in_c
        self.n_cross = n_cross
        
        self.Us = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, in_c, rank))) for _ in range(n_cross)])
        self.Vs = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, in_c, rank))) for _ in range(n_cross)])
        self.Cs = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, rank, rank))) for _ in range(n_cross)])
        self.gates = nn.ModuleList([nn.Linear(in_c, 1, bias=False) for _ in range(self.n_experts)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(in_c)) for _ in range(self.n_cross)])
        
        
    def forward(self, x):
        x0 = x
        xi = x
        
        for i in range(self.n_cross):
            gate_scores = []
            exp_outs = []
            for exp_id in range(self.n_experts):
                gate_scores.append(self.gates[exp_id](xi))
                
                v_x = torch.tanh(xi@self.Vs[i][exp_id])
                cv_x = torch.tanh(v_x@self.Cs[i][exp_id])
                ucv_x = cv_x@self.Us[i][exp_id].t() 
                expert_out = x0*(ucv_x + self.biases[i])
                exp_outs.append(expert_out)
                
            exp_outs = torch.stack(exp_outs, 2) # batch_size, in_c, n_expert
            gate_scores = torch.stack(gate_scores, 1) # batch_size, n_expert, 1
            moe_outs = torch.bmm(exp_outs, gate_scores).squeeze(-1)
            xi = moe_outs + xi
            
        return xi
    
        
class DCN(nn.Module):
    def __init__(
        self,
        dcn_params):
        super().__init__()
        
        self.ds_dim = dcn_params.get('ds_dim')
        self.sp_dim = dcn_params.get('sp_dim')
        self.num_cats = dcn_params.get('num_cats')
        self.embed_dim = dcn_params.get('embed_dim')
        self.hid_channels = dcn_params.get('hid_channels')
        self.cross_type = dcn_params.get('cross_type')
        self.n_cross = dcn_params.get('n_cross')
        self.dropout = dcn_params.get('dropout')
        self.concat_type = dcn_params.get('concat_type', 'parallel')
        self.rank = dcn_params.get('rank', 0)
        self.n_experts = dcn_params.get('n_experts', 0)
        
        input_dim = self.sp_dim * self.embed_dim + self.ds_dim
        
        # Initialize the Embedders
        self.embedders = nn.ModuleList()
        for i in range(self.sp_dim):
            self.embedders.append(nn.Embedding(self.num_cats[i], self.embed_dim))
            nn.init.xavier_uniform_(self.embedders[i].weight, gain=1)
        
        # Initialize the CrossNet
        if self.cross_type == 'vector':
            self.cross_net = CrossNetVector(input_dim, self.n_cross)
        elif self.cross_type == 'matrix':
            self.cross_net = CrossNetMatrix(input_dim, self.n_cross)
        elif self.cross_type == 'mix':
            if (self.n_experts > 0) & (self.rank > 0):
                self.cross_net = CrossNetMix(input_dim, self.n_cross, self.rank, self.n_experts)
            else:
                raise ValueError('Params `rank` and `n_experts` do not meet the require of CrossNet')
        else:
            raise ValueError(f"CrossNet use {self.cross_type} method is not define.")
        
        # Initialize DeepNet
        self.dnn = DNN([input_dim] + self.hid_channels, self.dropout)
        
        if self.concat_type == 'parallel':
            final_dim = input_dim + self.hid_channels[-1]
            self.final_linear = nn.Linear(final_dim, 1)
        elif self.concat_type == 'stacked':
            final_dim = self.hid_channels[-1]
            self.final_linear = nn.Linear(final_dim, 1)
        
    def forward(self, ds_input, sp_input):
        sp_embeds = []
        
        for i in range(self.sp_dim):
            sp_embeds.append(self.embedders[i](sp_input[:, i]))
        
        x = torch.cat(sp_embeds + [ds_input], dim=-1)
        
        x_cross = self.cross_net(x)
        if self.concat_type == 'parallel':
            x_dnn = self.dnn(x)
            x_cat = torch.cat([x_dnn, x_cross], dim=-1)
        elif self.concat_type == 'stacked':
            x_cat = self.dnn(x_cross)
        
        out = self.final_linear(x_cat)

        return torch.sigmoid(out)
        
        
        
        