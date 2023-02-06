import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split


def load_test_data(file_path):
    '''
    训练数据生成
    '''
    df = pd.read_csv(file_path, header=None, names=['text', 'target'])
    sentences, targets = df['text'].values, df['target'].values
    return train_test_split(sentences, targets)


def load_htl(path):
    review_df = pd.read_csv(path)
    # 去除unicode中文符号以外的内容
    review_df['review'] = review_df['review'].replace(['[^\u4e00-\u9fa5]+'], '', regex=True)
    review_df.dropna(inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(review_df['review'].values, # ndarray
                                                    review_df['label'].values,
                                                    train_size=0.7,
                                                    random_state=100)
    return X_train, X_test, y_train, y_test

class TextDataSet(Data.Dataset):
    def __init__(
        self,
        X,
        y) -> None:
        
        self.X = X
        self.y = y
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)