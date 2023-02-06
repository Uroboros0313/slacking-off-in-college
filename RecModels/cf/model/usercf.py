import pickle as pkl
from itertools import combinations
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


class UserCF():
    def __init__(
        self,
        df,
        ) -> None:
        pass
    
    def __init__(
        self,
        df,
        user_col='user',
        item_col='item',
        rate_col='rating',
        topN=10,
        K=10,
        n_jobs=8,
        iif=False,
        ) -> None:
        
        self.df = df
        self.user_col = user_col
        self.item_col = item_col
        self.rate_col = rate_col
        self.K = K
        self.topN = topN
        self.n_jobs = n_jobs
        self.iif = iif
    
        self.user_item_dict = dict()
        self.items = []
        self.users = []
        self.n_items = -1
        self.n_users = -1
        self.sim = None
        self.pop = None
        
    def init_data(self):
        df = self.df
        user_col = self.user_col
        item_col = self.item_col
        rate_col = self.rate_col
        
        items = list(df[item_col].unique())
        users = list(df[user_col].unique())
        item_user_dict = dict()
        user_item_dict = dict()
        sim = dict()
        pop = dict()
        
        print('Init Sim Table...')
        for user in users:
            sim[user] = defaultdict(int)
            pop[user] = 0
        
        print('Init Item-User Inverse Table...')
        '''
        Inverse Table
        '''
        for item in items:
            item_users = df[df[item_col] == item][user_col].to_list()
            ratings = df[df[item_col] == item][rate_col].to_list()
            item_user_dict[item] = dict(zip(item_users, ratings))
            
        for user in users:
            user_items = df[df[user_col] == user][item_col].to_list()
            ratings = df[df[user_col] == user][rate_col].to_list()
            user_item_dict[user] = dict(zip(user_items, ratings))
        
        self.items = items
        self.users = users
        self.sim = sim
        self.pop = pop
        self.item_user_dict = item_user_dict
        self.user_item_dict = user_item_dict
        self.n_items = len(items)
        self.n_users = len(users)
    
    @staticmethod
    def cal_item_attribute(iif, n_ius):
        if not iif:
            return 1
        else:
            return 1 / (np.log(1 + n_ius))
    
    def cal_similarity(self):
        iif = self.iif
        sim = self.sim
        users = self.users
        pop = self.pop
        item_user_dict = self.item_user_dict
        
        print('Calculating similarity...')
        for item, user_rates in tqdm(item_user_dict.items()):
            item_users = list(user_rates.keys())
            user_pair = combinations(item_users, 2)
            for user_1, user_2 in user_pair:
                score = self.cal_item_attribute(iif, len(item_users))
                sim[user_1][user_2] += score
                sim[user_2][user_1] += score
            
            for us in item_users:
                pop[us] += 1.0
                
        for u1, u2 in combinations(users, 2):
            sim[u1][u2] /= np.sqrt(pop[u1] * pop[u2])
            sim[u2][u1] /= np.sqrt(pop[u1] * pop[u2])
            
        self.sim = sim
        self.pop = pop
        
    def cal_user_rec(self, user, topN):
        vis_items = self.user_item_dict[user]
        user_sims = self.sim[user]
        
        rank = defaultdict(int)
        for user, user_sim in sorted(user_sims.items(), key=lambda x: x[1], reverse=True)[: topN]:
            for item, score in self.user_item_dict[user].items():
                if item in vis_items:
                    continue
                rank[item] += score * user_sim
        
        return rank
    
    def cal_recommend(self, users):
        topN = self.topN
        n_jobs = self.n_jobs
        
        print("Generating recommends...")
        ranks = Parallel(n_jobs=n_jobs, require='sharedmem')(
            delayed(self.cal_user_rec)(user, topN) for user in tqdm(users)
        )
        
        return ranks    