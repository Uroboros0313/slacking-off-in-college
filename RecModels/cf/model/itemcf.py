import pickle as pkl
from itertools import combinations
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


class ItemCF():
    def __init__(
        self,
        df,
        user_col='user',
        item_col='item',
        rate_col='rating',
        topN=10,
        K=10,
        n_jobs=8,
        iuf=False,
        norm=False,
        ) -> None:
        
        self.df = df
        self.user_col = user_col
        self.item_col = item_col
        self.rate_col = rate_col
        self.K = K
        self.topN = topN
        self.n_jobs = n_jobs
        self.iuf = iuf
        self.norm = norm
    
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
        user_item_dict = dict()
        sim = dict()
        pop = dict()
        
        print('Init Sim Table...')
        for item in items:
            sim[item] = defaultdict(int)
            pop[item] = 0
        
        print('Init User-Item Inverse Table...')
        '''
        《推荐系统实践》上基于倒排表计算相似度的做法
        '''
        for user in users:
            user_items = df[df[user_col] == user][item_col].to_list()
            ratings = df[df[user_col] == user][rate_col].to_list()
            user_item_dict[user] = dict(zip(user_items, ratings))
        
        self.items = items
        self.users = users
        self.sim = sim
        self.pop = pop
        self.user_item_dict = user_item_dict
        self.n_items = len(items)
        self.n_users = len(users)
    
    @staticmethod
    def cal_user_attribute(iuf, n_uit):
        if not iuf:
            return 1
        else:
            return 1 / np.log(1 + n_uit)

    def cal_similarity(self):
        user_item_dict = self.user_item_dict
        item_list = self.items
        sim = self.sim
        pop = self.pop
        iuf = self.iuf
        norm = self.norm
        
        # calculate similarity
        # sim是item-pair间对称的
        print('Calculating Similarity...')
        for _, user_rates in tqdm(user_item_dict.items()):
            user_items = list(user_rates.keys())
            item_pairs = combinations(user_items, 2)
            for item_1, item_2 in item_pairs:
                score = self.cal_user_attribute(iuf, len(user_items))
                sim[item_1][item_2] += score
                sim[item_2][item_1] += score
            
            for it in user_items:
                pop[it] += 1.0
        
        for item_1 in item_list:
            for item_2 in item_list:
                sim[item_1][item_2] /= np.sqrt(pop[item_1] * pop[item_2])
        
        if norm:
            for item in item_list:
                unorm_score = np.array(list(sim[item].values()))
                norm_sum = np.max(unorm_score)
                sim[item] = dict(zip(sim[item].keys(), unorm_score / norm_sum))
        
        self.sim = sim
        self.pop = pop
        
    
    def cal_user_rec(self, user, topN):
        K = self.K
        item_vis = self.user_item_dict[user]
        
        user_rank = defaultdict(int)
        
        '''
        对于每个不在user已访问列表item_vis的item——item_unvis
        计算该item的前K个相似性最高的item_nbr是否已被访问过
        如果是: 兴趣得分 interest = sim * user_rank
        '''
        
        for item in item_vis:
            it_sims = self.sim[item]
            for item_nbr, score in sorted(it_sims.items(), key=lambda x: x[1], reverse=True)[: K]:
                if item_nbr in item_vis.keys():
                    continue
                user_rank[item_nbr] += score * item_vis[item]
        
        user_rank = sorted(user_rank.items(), key=lambda x: x[1], reverse=True)[: topN]
        return user, user_rank
    
    def cal_recommend(self, users):
        n_jobs = self.n_jobs
        topN = self.topN
        
        users_rank = Parallel(n_jobs, require='sharedmem')(
            delayed(self.cal_user_rec)(user, topN) for user in tqdm(users) if user in self.users)
        
        return users_rank
        
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pkl.dump(self, f)