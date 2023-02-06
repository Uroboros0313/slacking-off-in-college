import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GCN, InnerProduct


class GAE(nn.Module):
    def __init__(
        self,
        in_c,
        hid_c1,
        dropout=0.2):
        super().__init__()
        
        self.gcn_1 = GCN(in_c, hid_c1, dropout)
        self.decoder = InnerProduct(dropout)
        
    def forward(self, x, adj):
        h = self.gcn_1(x, adj)
        mx = self.decoder(h)
        return mx


class VGAE(nn.Module):
    def __init__(
        self,
        in_c,
        hid_c1,
        hid_c2,
        dropout=0.2):
        super().__init__()
        
        self.gcn_1 = GCN(in_c, hid_c1, dropout)
        self.gcn_mu = GCN(hid_c1, hid_c2, dropout)
        self.gcn_var = GCN(hid_c1, hid_c2, dropout)
        self.decoder = InnerProduct(dropout)
        
    def forward(self, x, adj):
        mu, log_var = self.encode(x, adj)
        z = self.sampling(mu, log_var)
        mx = self.decoder(z)
        return mx
    
    def encode(self, x, adj):
        h = self.gcn_1(x, adj)
        mu = self.gcn_mu(h, adj)
        log_var = self.gcn_var(h, adj)
        
        return mu, log_var
    
    def sampling(self, mu, log_var):
        if self.training:
            std = torch.exp(log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    @staticmethod
    def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        BCE = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return BCE + KLD
        
        

       