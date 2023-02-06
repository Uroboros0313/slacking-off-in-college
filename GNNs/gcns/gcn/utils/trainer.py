import torch

from .data_utils import accuracy

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        n_epochs,
        dataset,
        device):
        
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.features = dataset['features'].to(device)
        self.adj = dataset['adj'].to(device)
        self.labels = dataset['labels'].to(device)
        self.idx_train = dataset['idx_train']
        self.idx_val = dataset['idx_val']
        self.idx_test = dataset['idx_test']
        
    def train(self):
        features = self.features
        labels = self.labels
        adj = self.adj
        
        for epoch in range(self.n_epochs):
            self.model.train()
            l = self._train(features, labels, adj)
            print('Epoch: {} || Train Loss: {}'.format(epoch, l))
            
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    self._eval(features, labels, adj, 'train')
                    self._eval(features, labels, adj, 'val')
        
        self._eval(features, labels, adj, 'test')
            
    def _train(self, features, labels, adj):
        out = self.model(features, adj)
        train_out = out[self.idx_train]
        train_y = labels[self.idx_train]
        
        l = self.criterion(train_out, train_y)
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        
        return l.item()
    
    def _eval(self, features, labels, adj, evalset_name='val'):
        idx = getattr(self, f'idx_{evalset_name}')
        
        out = self.model(features, adj)
        eval_out = out[idx]
        eval_y = labels[idx]
        
        l = self.criterion(eval_out, eval_y)
        print('{} Loss: {} || Acc: {} || '.format(evalset_name.upper(), l.item(), accuracy(eval_out, eval_y)))
        return 

