import torch
import torch.optim as optim
import torch.nn as nn

from utils.data_utils import *
from utils.trainer import Trainer
from model.gcn import GCN

DATA_PATH = "../data/cora/"
DATASET = 'cora'
IN_C = 1433
HID_C = 16
OUT_C = 7
N_EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__=='__main__':
    dataset = load_data(DATA_PATH, DATASET)
    model = GCN(IN_C, HID_C, OUT_C, DROPOUT)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), LR, weight_decay=WEIGHT_DECAY)    
    trainer = Trainer(model, optimizer, criterion, N_EPOCHS, dataset, DEVICE)
    trainer.train()