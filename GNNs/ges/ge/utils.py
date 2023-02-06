import random
import itertools

from joblib import Parallel, delayed
import numpy as np


class DataGenerator():
    def __init__(
        self,
        seqs,
        window_size,
        node2idx) -> None:
        self.seqs = seqs
        self.window_size = window_size
        self.node2idx = node2idx
        
    @staticmethod
    def make_pairs(seq, window_size, node2idx):
        pairs = []
        for idx in range(len(seq)):
            w_size = random.choice(range(1, window_size))
            for w_idx in range(max(0, idx - w_size), min(len(seq), idx + w_size)):
                if seq[w_idx] != seq[idx]:
                    pairs.append([node2idx[seq[idx]], node2idx[seq[w_idx]]])
        return pairs
    
    def make_data(self, n_jobs=1):
        pairs_list = Parallel(n_jobs=n_jobs, require='sharedmem')(
            delayed(self.make_pairs)(seq, self.window_size, self.node2idx) for seq in self.seqs
        )
        pairs_list = list(itertools.chain(*pairs_list))
        return np.array(pairs_list)


def create_alias_table(area_ratio):
    
    small, large= [], []
    l = len(area_ratio)
    _area_ratio = np.array(area_ratio) * l
    
    for i, prob in enumerate(_area_ratio):
        if prob < 1:
            small.append(i)
        else:
            large.append(i)
    
    accept, alias = [0] * l, [0] * l   
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = _area_ratio[small_idx]
        alias[small_idx] = large_idx
        # large_idx的元素拼接进small_idx的元素后, 计算large_idx对应的剩余的概率块
        _area_ratio[large_idx] =\
            _area_ratio[large_idx] - (1 - _area_ratio[small_idx])
            
        if _area_ratio[large_idx] > 1:
            large.append(large_idx)
        else:
            small.append(large_idx)
            
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
        
    return accept, alias


def alias_sample(accept, alias):
    
    N = len(accept)
    k = int(np.random.random() * N)
    r = np.random.random()
    if accept[k] < r:
        return alias[k]
    else:
        return k
        
        