import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from sklearn.metrics import roc_auc_score, accuracy_score


def train(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device, enabled_amp=False):
    
    enabled_amp = enabled_amp&(device == 'cuda')
        
    for epoch in range(n_epochs):
        l_sum = 0.0
        start = time.time()
        model.train()
        for i, (X, y) in enumerate(train_loader):
            grad_scaler = amp.GradScaler(enabled=enabled_amp)
            with amp.autocast(enabled=enabled_amp):
                y = torch.tensor(y, dtype=torch.float).reshape(-1, 1).to(device)
                out = model(X)
                l = criterion(out, y)
                optimizer.zero_grad()
                
                grad_scaler.scale(l).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            
        l_sum += l.item()
        end = time.time()
        
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        lr = optimizer.param_groups[0]['lr']
        print('Epoch: {} || GPU Occupy: {:.4f}.MiB || Train Loss: {:.4f} || LR: {:.4f} || Epoch time: {:.4f} ||'.\
            format(epoch, gpu_mem_alloc, l_sum, lr, end - start))
        evaluate(model, valid_loader)
        
    return model

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        preds = []
        ys = []
        for X, y in loader:
            pred = model.predict(X)
            preds.append(pred.cpu().numpy())
            ys.append(y)
            
        preds = np.concatenate(preds, axis=0)
        ys = np.concatenate(ys, axis=0)
        
    print('VALID || AUC: {} || ACC: {}'.format(
        roc_auc_score(ys, preds), accuracy_score(ys, np.round(preds, 0))
        ))
        
            
        
        