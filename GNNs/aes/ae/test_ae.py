import torch
import torch.nn as nn
import torch.optim as optim
from pyod.utils.data import generate_data
from sklearn.metrics import accuracy_score, recall_score

from model import AE


IN_C = 50
HID_C = 16
NUM_TRAIN = 10000
NUM_TEST = 2000
LR = 0.001
EPOCHS = 20

X_train, X_test, y_train, y_test =\
    generate_data(
        n_train=NUM_TRAIN, 
        n_test=NUM_TEST, 
        n_features=IN_C, 
        contamination=0.2, 
        behaviour='new')

print('|| Train: X shape {}, y shape {}|| Test: X shape {}, y shape {}||'.\
    format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

trn_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
trn_iter = torch.utils.data.DataLoader(trn_ds, batch_size=32, )


def train(model, train_iter, epochs, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        l_sum = 0.0
        for i, (X, y) in enumerate(train_iter):
            X_pos = X[y == 0]
            if len(X_pos) <= 0:
                continue
            pos_out = model(X_pos)
            l = criterion(pos_out, X_pos)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    
            l_sum += l.item()
        print("EPOCH {} || TRAIN LOSS :{}".format(epoch, l_sum))
    return model


def mse_loss(y_pred, y_true):
    loss = torch.sum((y_pred - y_true)**2, dim=-1)
    return loss


def select_threshold(model, X_train, y_train):
    model.eval()
    with torch.no_grad():
        X, y = torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)
        X_out = model(X)
        divs = mse_loss(X_out, X)
        threshold = max(divs[y == 0])
    
    return threshold.item()


def detect_anomaly(model, threshold, X):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        divs = mse_loss(X, pred)
        labels = (divs > threshold).to(torch.int)
        
    return labels

    
if __name__ == "__main__":
    model = AE(in_c=IN_C, hid_c=HID_C)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    model = train(model, trn_iter, EPOCHS, optimizer, criterion)
    
    threshold = select_threshold(model, X_train, y_train)
    
    X_test, y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.int)
    res = detect_anomaly(model, threshold, X_test)
    
    acc = accuracy_score(y_test.numpy(), res.numpy())
    recall = recall_score(y_test.numpy(), res.numpy())
    print('acc: {}, recall: {}'.format(acc, recall))
    
       