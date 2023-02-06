import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device='cpu'):
    for epoch in range(n_epochs):
        model.train()
        l_sum = 0.0
        for *inputs, y in train_loader:
            model_inputs = [input.to(device) for input in inputs]
            y = y.to(device)
            out = model(*model_inputs)
            optimizer.zero_grad()
            l = criterion(out, y)
            l = model.cal_weight_reg(l)
            l.backward()
            optimizer.step()
            
            l_sum += l.item()
        
        if (epoch + 1) % 5 == 0:
            print("EPOCH {}: TRAIN LOSS: {}".format(epoch + 1, l_sum))
            evaluate(model, train_loader, 'TRAIN', device)
            evaluate(model, valid_loader, device = device)
    return model
            
            
def evaluate(model, loader, ds_name='VALID', device='cpu'):
    model.eval()
    with torch.no_grad():
        ys = []
        preds = []
        for *inputs, y in loader:
            model_inputs = [input.to(device) for input in inputs]
            y = y.to(device)
            out = model(*model_inputs)
            preds.append(out)
            ys.append(y)
            
        preds = torch.cat(preds, dim=0).cpu().numpy()
        ys = torch.cat(ys, dim=0).cpu().numpy()
    
    print("{} || AUC: {} || ACC: {} ||".format(ds_name, *cal_metrics(preds, ys)))
        

def cal_metrics(preds, ys):
    
    auc = roc_auc_score(ys, preds)
    acc = accuracy_score(ys, np.round(preds, decimals=0))
    return auc, acc 
    