import torch
import torch.optim as optim
import torch.nn as nn

from utils.data_utils import *
from utils.trainer import Trainer
from model.gat import GATv1, GATv2

MODEL = 'gatv1'
DATA_PATH = "../data/cora/"
DATASET = 'cora'
IN_C = 1433
HID_C = 8
OUT_C = 7
N_EPOCHS = 300
LR = 0.005
DROPOUT = 0.6
ALPHA = 0.2
NHEADS = 8
WEIGHT_DECAY = 5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__=='__main__':
    dataset = load_data(DATA_PATH, DATASET)
    if MODEL == 'gatv1':
        model = GATv1(IN_C, HID_C, OUT_C, DROPOUT, ALPHA, NHEADS,)
    else:
        model = GATv2(IN_C, HID_C, OUT_C, DROPOUT, ALPHA, NHEADS)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), LR, weight_decay=WEIGHT_DECAY)    
    trainer = Trainer(model, optimizer, criterion, N_EPOCHS, dataset, DEVICE)
    trainer.train()