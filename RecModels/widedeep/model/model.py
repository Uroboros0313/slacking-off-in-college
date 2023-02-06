import torch
import torch.nn as nn

class Wide(nn.Module):
    def __init__(
        self,
        in_c,
        ):
        super().__init__()
        self.in_c = in_c
        self.fc = nn.Linear(in_c, 1)
    
    def forward(self, x):
        return self.fc(x)

class Deep(nn.Module):
    def __init__(
        self,
        in_c,
        hid_channels
        ):
        super().__init__()
        
        channels = [in_c] + hid_channels
        fcs = nn.ModuleList()
        for i in range(len(channels) - 1):
            fcs.append(nn.Linear(channels[i], channels[i + 1]))
            fcs.append(nn.ReLU())
        self.net = nn.Sequential(*fcs)
        
    def forward(self, x):
        return self.net(x)
        

class WideDeep(nn.Module):
    def __init__(
        self,
        wd_params,
        ):
        super().__init__()
        
        embed_dim = wd_params.get("embed_dim", 5)
        hid_channels = wd_params.get("hid_channels", [256, 128, 64])
        cross_dim = wd_params.get("cross_dim")
        sp_dim = wd_params.get("sp_dim")
        ds_dim = wd_params.get("ds_dim")
        num_cats = wd_params.get("num_cats")
        
        embeders = nn.ModuleList()
        for i in range(sp_dim):
            embeders.append(nn.Embedding(num_cats[i], embed_dim))
            
        wide_dim = cross_dim + ds_dim
        deep_dim = sp_dim * embed_dim + ds_dim
        
        self.embeders = embeders
        self.wide_part = Wide(in_c=wide_dim)
        self.deep_final = nn.Linear(hid_channels[-1], 1)
        self.deep_part = Deep(in_c=deep_dim, hid_channels=hid_channels)
        self.embed_dim = embed_dim
        self.hid_channels = hid_channels
        self.cross_dim = cross_dim
        self.sp_dim = sp_dim
        self.ds_dim = ds_dim
        self.num_cats = num_cats
        
    def forward(self, ds_input, cross_input, sp_input):
        
        deep_input = []
        for i in range(self.sp_dim):
            deep_input.append(self.embeders[i](sp_input[..., i]))
        deep_input.append(ds_input)
        deep_input = torch.cat(deep_input, dim=-1)
        
        wide_input = [ds_input, cross_input]
        wide_input = torch.cat(wide_input, dim=-1)
        
        deep_out = self.deep_part(deep_input)
        deep_out = self.deep_final(deep_out)
        wide_out = self.wide_part(wide_input)
        assert (wide_out.shape == deep_out.shape)
        
        return torch.sigmoid(0.5 * (wide_out + deep_out))
        
        