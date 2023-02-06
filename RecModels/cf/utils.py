import pandas as pd


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            user, item, _, __ = line.strip().split()
            data.append([user, item, 1.0])
    return data

def read_dataframe(filename, sep=',', header=None, names=[]):
    df = pd.read_csv(filename, sep=sep, header=header, names=names)
    return df