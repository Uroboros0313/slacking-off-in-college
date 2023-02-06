from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(
        self,
        train:pd.DataFrame,
        test:pd.DataFrame,
        id_col,
        label_col) -> None:
        
        self.train = train
        self.test = test
        self.id_col = id_col
        self.label_col = label_col
        
        self.enc_dict = dict()
        self.sp_counts = dict()
        self.mms_dict = dict()
        self.cross_map = dict()
        self.cross_cols = None
        self.ds_cols = None
        self.sp_cols = None
        self.train_processed = None
        self.valid_processed = None
        self.test_processed = None
        
    @staticmethod
    def label_enc(series, ):
        lb_enc = LabelEncoder()
        enc_ss = lb_enc.fit_transform(series)
        return enc_ss, enc_ss.max() + 1 , lb_enc
    
    @staticmethod
    def minmax_transform(series, ):
        scaler = MinMaxScaler()
        mms_ss = scaler.fit_transform(series.values.reshape(-1, 1))
        return mms_ss, scaler
    
    @staticmethod
    def cross_product_pair(ss1, ss2):
        c1 = ss1.unique()
        c2 = ss2.unique()
        
        combs = list(product(c1, c2))
        combs_id = range(len(combs))
        map_ = dict(zip(combs, combs_id))
        ss_comb = pd.Series(zip(ss1, ss2)).map(map_)
        
        return ss_comb, map_

    def data_process(self, cross_product_pairs=[]):
        train = self.train.copy()
        test = self.test.copy()
        id_col = self.id_col
        enc_dict = self.enc_dict
        mms_dict = self.mms_dict
        sp_counts = self.sp_counts
        cross_map = self.cross_map
        
        df = pd.concat([train, test]).reset_index(drop=True)
        del df[id_col]

        feats = df.columns
        sp_cols = feats[df.columns.str.startswith('C')].to_list()
        ds_cols = feats[df.columns.str.startswith('I')].to_list()
        cross_cols = []
        print("DataFrame Contains Label: \n|| {} ||".format(self.label_col))
        print("DataFrame Contains Sparse Features: \n|| {} ||".format(sp_cols))
        print("DataFrame Contains Dense Features: \n|| {} ||".format(ds_cols))

        # 填充缺失值
        print(">>> Process Nan Values...")
        df[ds_cols] = df[ds_cols].fillna(df[ds_cols].mean()) # 稠密特征均值填充
        df[sp_cols] = df[sp_cols].fillna('<UNKNOWN>') # 稀疏特征填充为统一的未知类别
        
        # 稀疏特征交叉积
        cross_cols = []
        print(">>> Make Cross Product...")
        for col1, col2 in cross_product_pairs:
            if (col1 not in sp_cols)|(col2 not in sp_cols):
                raise ValueError("corss product col-pair should all in `sp_cols`.")

            ss_comb, cross_map[f"CP_{col1}_{col2}"] =\
                self.cross_product_pair(df[col1], df[col2])
            df = pd.concat([df, pd.get_dummies(ss_comb, prefix=f"CP_{col1}_{col2}")], axis=1)
            new_cols = df.columns[df.columns.str.startswith(f"CP_{col1}_{col2}")]
            cross_cols.extend(new_cols)
        
        # 稀疏特征标签编码
        print(">>> Sparse Feature Encoding...")
        for col in sp_cols:
            df[col], sp_counts[col], enc_dict[col] = self.label_enc(df[col])
        
        # 数值型特诊归一化
        print(">>> Dense Feature Transform...")
        for col in ds_cols:
            df[col], mms_dict[col] = self.minmax_transform(df[col])
        
        train, test = df.iloc[: train.shape[0],: ], df.iloc[train.shape[0]: ,: ]
        train, valid = train_test_split(train, test_size=0.2)
        
        self.train_processed = train.reset_index(drop=True)
        self.valid_processed = valid.reset_index(drop=True)
        self.test_processed = test.reset_index(drop=True)
        self.sp_cols = sp_cols
        self.ds_cols = ds_cols
        self.cross_cols = cross_cols
        self.enc_dict = enc_dict
        self.mms_dict = mms_dict
        self.cross_map = cross_map
        self.sp_counts = sp_counts
        
    def get_dataset(self, data_type='processed'):
        if data_type == 'raw':
            train = self.train_processed.copy()
            test = self.valid_processed.copy()
            return train, test
    
        elif data_type == 'processed':
            if isinstance(self.train_processed, pd.DataFrame):
                train = self.train_processed.copy()
                valid = self.valid_processed.copy()
                test = self.valid_processed.copy()
                return train, valid, test
            
            else:
                raise ValueError("Process data before call < data_type=`processed` >.")
    
    def get_Xy_dataset(self, dataset='train'):
        
        label_col = self.label_col
        
        if dataset=='train':
            ds = self.train_processed.copy()
        elif dataset=='valid':
            ds = self.valid_processed.copy()
        elif dataset=='test':
            ds = self.test_processed.copy()
            
        y = ds[label_col]
        del ds[label_col]
        X = ds
        
        return X, y
    
    def get_tensor_dataset(self, dataset='train'):
        ds_cols = self.ds_cols
        cross_cols = self.cross_cols
        sp_cols = self.sp_cols
        X, y = self.get_Xy_dataset(dataset)
        
        ds_input = X[ds_cols].values
        cross_input = X[cross_cols].values
        sp_input = X[sp_cols].values
        y = y.values
        
        
        ds_input = torch.tensor(ds_input, dtype=torch.float)
        cross_input = torch.tensor(cross_input, dtype=torch.float)
        sp_input = torch.tensor(sp_input, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
        
        dataset = TensorDataset(ds_input, cross_input, sp_input, y)
        return dataset
    
    def get_loader(self, dataset='train', batch_size=32, shuffle=False):
        dataset = self.get_tensor_dataset(dataset=dataset)
        loader = DataLoader(dataset, batch_size, shuffle)
        return loader
            