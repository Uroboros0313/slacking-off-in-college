import pandas as pd
import torch


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            user, item, _, __ = line.strip().split()
            data.append([user, item, 1.0])
    return data


def read_dataframe(filename, sep=',', header=None, names=[], encoding='utf-8'):
    df = pd.read_csv(filename, sep=sep, header=header, names=names, encoding=encoding)
    return df


def get_dummy_df(df, dummy_cols=[]):
    if len(dummy_cols) > 0:
        for col in dummy_cols:
            dm_feature = pd.get_dummies(df[col])
            df = pd.concat([df, dm_feature], axis=1)
            df = df.drop(col, axis=1)

    return df


def get_tensor_dataset(X, y, batch_size=32, shuffle=False):
    in_shape = X.shape
    out_shape = y.shape
    
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    
    if len(in_shape) == 1:
        X = X.reshape(-1, 1)
    if len(out_shape) == 1:
        y = y.reshape(-1, 1)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader