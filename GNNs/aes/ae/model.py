import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(
        self,
        in_c,
        hid_c,
        ):
        super().__init__()
        
        self.in_c = in_c
        self.hid_c = hid_c
        
        self.encoder = nn.Linear(in_c, hid_c)
        self.decoder = nn.Linear(hid_c, in_c)
        
    def forward(self, x):
        x_encode = F.relu(self.encoder(x))
        x_decode = self.decoder(x_encode)
        return x_decode


class VAE(nn.Module):
    def __init__(
        self, 
        in_c, 
        e_hid_c1,
        e_hid_c2, 
        distr_dim, 
        d_hid_c1,
        d_hid_c2,):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_c, e_hid_c1),
            nn.ReLU(),
            nn.Linear(e_hid_c1, e_hid_c2),
            nn.ReLU()
        )
        self.mu_fc = nn.Linear(e_hid_c2, distr_dim)
        self.log_var_fc = nn.Linear(e_hid_c2, distr_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(distr_dim, d_hid_c1),
            nn.ReLU(),
            nn.Linear(d_hid_c1, d_hid_c2),
            nn.ReLU(),
            nn.Linear(d_hid_c2, in_c)
        )
        
    def forward(self, x):
        mu, log_var = self.encode_input(x)
        z = self.sampling(mu, log_var)
        out = self.decode_sample(z)
        return out, mu, log_var
    
    def encode_input(self, x):
        '''
        return:
        - mu: approximate avg
        - log_var: approximate log of std
        '''
        h = self.encoder(x)
        mu, log_var = self.mu_fc(h), self.log_var_fc(h)
        return mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(log_var * 0.5) # inverse transform to std
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode_sample(self, z):
        return F.sigmoid(self.decoder(z))