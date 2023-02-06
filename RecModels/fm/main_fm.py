from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_tensor_dataset
from model.fm import FMModel


BATCH_SIZE = 32
LR = 0.01
N_EPOCHS = 50
DATA_DIR = Path('../data/user_data')

df = pd.read_csv(DATA_DIR / 'fm_data.csv')
y = df['rating']
del df['rating']
X = df

n = X.shape[1]
k = 5

if __name__ == '__main__':
    trn_x, tst_x, trn_y, tst_y = train_test_split(X, y, test_size=0.2,)
    train_loader = get_tensor_dataset(trn_x.values, trn_y.values, BATCH_SIZE)
    test_loader = get_tensor_dataset(tst_x.values, tst_y.values, BATCH_SIZE)
    model = FMModel(n, k)
    model.fit(train_loader, n_epochs=N_EPOCHS, lr=LR)
    model.eval(train_loader)
    model.eval(test_loader)