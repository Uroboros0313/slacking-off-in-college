import pathlib
import sys
sys.path.append("..")

import torch
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from ge import Node2Vec


DATA_PATH = pathlib.Path('../dataset/wiki/Wiki_edgelist.txt')
WINDOW_SIZE = 5
WALK_LEN = 10
NUM_WALK = 20
EMBED_DIM = 128
BATCH_SIZE = 1024
EPOCHS = 1
N_JOBS = 8
P = 0.25
Q = 4
USE_REJECTION_SAMPLING = 0
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


G = nx.read_edgelist(
    DATA_PATH,
    create_using=nx.DiGraph(), 
    nodetype=None, 
    data=[('weight', int)])
print('number of edges: {}'.format(G.number_of_edges()))
print('number of nodes: {}'.format(G.number_of_nodes()))


def read_node_label(filename):
    df = pd.read_csv(filename, sep=' ', names=['node', 'label'])
    return df['node'].values, df['label'].values


def visualize_tsne(model, X, y):
    embed_dict = model._get_embeddings()
    embs = []
    for node in X:
        embs.append(embed_dict[str(node)])
        
    tsne = TSNE(n_components=2)
    embs_2d = tsne.fit_transform(embs)
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], s=1, c=y)
    plt.show()
    
    
if __name__ == '__main__':
    n2v = Node2Vec(
        G,
        EMBED_DIM,
        WALK_LEN,
        NUM_WALK,
        P,
        Q,
        USE_REJECTION_SAMPLING,
        WINDOW_SIZE,
        N_JOBS,
        DEVICE,
    )

    n2v.init_dataset(BATCH_SIZE)
    n2v.train(EPOCHS, lr=0.025)
    
    X, y = read_node_label('../dataset/wiki/wiki_labels.txt')
    visualize_tsne(n2v, X, y)

