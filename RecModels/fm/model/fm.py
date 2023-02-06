import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

class FM(nn.Module):
    def __init__(
        self,
        n,
        k,
        ):
        super().__init__()
        
        self.w0 = nn.Parameter(torch.randn((1)), requires_grad=True)
        self.w = nn.Linear(n, 1)
        self.v = nn.Parameter(torch.randn((n, k)), requires_grad=True)
        
    def forward(self, x):
        
        lin = self.w(x)
        cross_1 = torch.mm(x, self.v)
        cross_2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        out = self.w0 + lin +\
            0.5 * torch.sum((torch.pow(cross_1, 2) - cross_2), dim=1, keepdim=True)
        return out
    
    
class FMModel():
    def __init__(
        self,
        n,
        k=5
        ) -> None:
        
        self.n = n
        self.k = k
        self.model = FM(n, k)
        
    def fit(
        self,
        train_loader,
        n_epochs=100,
        lr=0.001,
        loss_func='bce',
        optim_method='adam'):
        
        model = self.model
        if loss_func == 'bce':
            criterion = nn.BCEWithLogitsLoss() # 记一下BCE和withlogit的区别
        if optim_method=='adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            l_sum = 0.0
            model.train()
            for X, y in train_loader:
                out = model(X)
                l = criterion(out, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                
                l_sum += l.item()
            
            self.model = model
            if (epoch + 1) % 10 ==0:    
                print("EPOCH: {} || TRAIN LOSS: {:.4f}".format(epoch + 1, l_sum))
        
    def eval(self, test_loader):
        model = self.model
        model.eval()
        with torch.no_grad():
            preds = []
            ys = []
            for X, y in test_loader:
                out = model(X)
                preds.append(out)
                ys.append(y)
            
            preds = torch.cat(preds, dim=0).cpu().numpy()
            ys = torch.cat(ys, dim=0).cpu().numpy()
        
        print("TEST RESULT || AUC: {} ||".format(self.eval_metric(ys, preds)))
                
    @staticmethod
    def eval_metric(y, pred):
        auc = roc_auc_score(y, pred)
        return auc
                