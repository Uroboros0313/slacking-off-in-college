import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from model.model import BertChineseClassifier
from utils.data_utils import TextDataSet, load_htl
from trainer import train
from configs.sentiment_htl import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENABLED_AMP = True
# w/ : 3379.9480.MiB 
# w/o: 3530.9430.MiB
print('torch using {} to train model...'.format(DEVICE))

X_train, X_test, y_train, y_test = load_htl(DATA_PATH)
train_set, test_set = TextDataSet(X_train, y_train), TextDataSet(X_test, y_test)
train_loader, test_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE), Data.DataLoader(test_set, batch_size=BATCH_SIZE)
print("num train samples: {}, num test samples: {}".format(len(train_set), len(test_set)))


if __name__=='__main__':
    model = BertChineseClassifier(MODEL_PATH, freeze=True, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    model = train(model, train_loader, test_loader, optimizer, criterion, N_EPOCHS, DEVICE, ENABLED_AMP)
    