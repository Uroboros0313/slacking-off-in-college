import random

from itertools import chain
from joblib import Parallel, delayed

from .utils import create_alias_table, alias_sample


class RandomWalker:
    def __init__(
        self,
        graph,
        walk_len,
        num_walk,
        n_jobs=-1,
        walk_method='deepwalk',
        p=1,
        q=1,
        use_rejection_sampling=0,
        ):
        self.graph = graph
        self.n_jobs = n_jobs
        self.walk_len = walk_len
        self.num_walk = num_walk
        self.walk_method = walk_method
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling
        
        self.alias_edges = None
        self.alias_nodes = None
    
    def random_walk(self, start_node):
        graph = self.graph
        walk_len = self.walk_len
        walk = [start_node]
        
        while len(walk) < walk_len:
            cur_node = walk[-1]
            nbrs = list(graph.neighbors(cur_node))
            
            if len(nbrs) > 0:
                walk.append(random.choice(nbrs))
            else:
                break
        return walk
    
    def policy_walk(self, start_node):
        graph = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk_len = self.walk_len
        
        walk = [start_node]
        
        while len(walk) < walk_len:
            cur = walk[-1]
            cur_nbrs = list(graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break
            
        return walk
    
    def _gen_walks(self, nodes):
        node_walks = []
        
        if self.walk_method == 'deepwalk':
            walk_opt = self.random_walk
        elif self.walk_method == 'node2vec':
            walk_opt = self.policy_walk
        
        for node in nodes:
            node_walks.append(walk_opt(node))
            
        return node_walks
    
    def gen_walks(self):
        nodes = list(self.graph.nodes())
        
        walks = Parallel(n_jobs=self.n_jobs, require='sharedmem')(
            delayed(self._gen_walks)(nodes) for _ in range(self.num_walk))
        walks = list(chain(*walks))
        
        return walks
    
    def process_trans_prob(self):
        graph = self.graph
        
        alias_nodes = dict()
        for node in graph.nodes():
            alias_nodes[node] =self.get_alias_node(node)
        self.alias_nodes = alias_nodes
        
        if not self.use_rejection_sampling:
            alias_edges = dict()
            for edge in graph.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                if not graph.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
            
            self.alias_edges = alias_edges
        
        return
    
    def get_alias_node(self, node):
        
        graph = self.graph
        
        unnorm_probs = [graph[node][nbr].get('weight', 1.0) for nbr in graph.neighbors(node)]
        sum_probs = sum(unnorm_probs)
        norm_probs = [float(unnorm_prob) / sum_probs for unnorm_prob in unnorm_probs]
        
        return create_alias_table(norm_probs)
        
    def get_alias_edge(self, t, v):
        
        graph = self.graph
        p, q = self.p, self.q
        
        unnorm_probs = []
        for x in self.graph.neighbors(v):
            weight = graph[v][x].get('weight', 1.0)
            if x == t:
                unnorm_prob = weight / p
            elif graph.has_edge(x, t):
                unnorm_prob = weight
            else:
                unnorm_prob = weight / q
            
            unnorm_probs.append(unnorm_prob)        
        
        sum_probs = sum(unnorm_probs)
        norm_probs = [prob / sum_probs for prob in unnorm_probs]
        
        return create_alias_table(norm_probs)        