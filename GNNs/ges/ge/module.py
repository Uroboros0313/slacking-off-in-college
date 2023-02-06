import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
    # TODO: implement negtive sampling and hsoftmax
    def __init__(
        self,
        vocab_size,
        embed_dim):
        super().__init__()
        
        self.W = nn.Parameter(torch.empty((vocab_size, embed_dim)), requires_grad=True)
        nn.init.xavier_uniform_(self.W.data)
        self.V = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.W[x, :]
        x = self.V(x)
        return F.softmax(x, dim=1)