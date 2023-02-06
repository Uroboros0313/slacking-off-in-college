from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data_utils import DataProcessor
from utils import trainer
from model.model import NFM


DATA_DIR = Path('../data/criteo/origin_data/')
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
LABEL = 'Label'
ID = 'Id'
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 0.0005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


train = pd.read_csv(DATA_DIR / TRAIN_FILE)
test = pd.read_csv(DATA_DIR / TEST_FILE)

data_processor = DataProcessor(train, test, ID, LABEL)
data_processor.data_process()

loader = dict()
for ds in ['train', 'valid', 'test']:
    loader[ds] = data_processor.get_loader(ds, BATCH_SIZE, )
    
nfm_params = {
    "hid_channels": [256, 128, 64],
    "num_cats": list(data_processor.sp_counts.values()),
    "sp_dim": len(data_processor.sp_cols),
    "ds_dim": len(data_processor.ds_cols),
    "embed_dim": 5,
    "bi_dropout": 0.2,
}

# TODO: implements the weight reg
if __name__ == '__main__':
    model = NFM(nfm_params).to(DEVICE)
    train_loader, valid_loader, test_loader = loader['train'], loader['valid'], loader['test']

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model = trainer.train(model, train_loader, valid_loader, criterion, optimizer, N_EPOCHS, DEVICE)