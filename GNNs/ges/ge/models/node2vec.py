from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from ..walker import RandomWalker
from ..module import Word2Vec
from ..utils import DataGenerator


class Node2Vec:
    def __init__(
        self,
        G,
        embed_dim,
        walk_len,
        num_walk,
        p,
        q,
        use_rejection_sampling,
        window_size,
        n_jobs,
        device
        ) -> None:
        
        self.G = G
        self.embed_dim = embed_dim
        self.walk_len = walk_len
        self.num_walk = num_walk
        self.device = device
        self.n_jobs = n_jobs
        self.window_size = window_size
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling
        
        self.walker = None
        self.model = None
        self.node2idx = None
        self.iter = None
        
    def train(self, epochs, lr=0.0025):
        self.model = Word2Vec(vocab_size=len(self.G.nodes()), embed_dim=self.embed_dim)
        self.model.to(self.device)
        
        optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in tqdm(range(epochs)):
            l_sum = 0.0
            for X, y in self.iter:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                l = criterion(out, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                
                l_sum += l.item()
            
            print('Epoch {} >>> Train Loss: {}'.format(epoch + 1, l_sum))
            
    def init_dataset(self, batch_size):
        walker = RandomWalker(
            graph = self.G,
            walk_len = self.walk_len, 
            num_walk = self.num_walk,
            n_jobs = self.n_jobs,
            p=self.p,
            q=self.q,
            use_rejection_sampling=self.use_rejection_sampling,
            walk_method='node2vec'
        )
        walker.process_trans_prob()
        seqs = walker.gen_walks()
        self.walker = walker
        
        nodes = list(self.G.nodes)
        node_idxs = list(range(len(self.G.nodes)))
        self.node2idx = dict(zip(nodes, node_idxs))
        self.idx2node = dict(zip(node_idxs, nodes))
        data_generator = DataGenerator(
            seqs, 
            self.window_size,
            self.node2idx,
        )
        dataset = data_generator.make_data(self.n_jobs)
        
        X, y = torch.tensor(dataset[:, 0], dtype=torch.long), torch.tensor(dataset[:, 1], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
        data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
        self.iter = data_iter
        return data_iter
    
    def _get_embeddings(self):
        _embeddings = self.model.W.data.cpu().numpy()
        _embed_dict = dict()
        for i in range(len(_embeddings)):
            _embed_dict[self.idx2node[i]] = _embeddings[i, :]
        return _embed_dict