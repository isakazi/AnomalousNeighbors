from argparse import ArgumentError
import os
import pickle

import torch
from torch_geometric.utils import dense_to_sparse, to_undirected, to_dense_adj
from torch_geometric.data import Data

from .load_utils import edge_tv_split, edge_tvt_split


def std_edge_w(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        print(ew_t)

        ew_t = (ew_t.long() / ew_t.std())  # .long()
        print(ew_t)
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)
    return ews


def normalized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        ew_t = ew_t.true_divide(ew_t.mean())
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews


def standardized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        std = ew_t.std()

        # Avoid div by zero
        if std.item() == 0:
            ews.append(torch.full(ew_t.size(), 0.5))
            continue

        ew_t = (ew_t - ew_t.mean()) / std
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews


class TData(Data):
    TR = 0
    VA = 1
    TE = 2
    ALL = 3

    def __init__(self, **kwargs):
        super(TData, self).__init__(**kwargs)


        self.tr = lambda t: self.eis[t][:, self.masks[t][0]]
        self.va = lambda t: self.eis[t][:, self.masks[t][1]]
        self.te = lambda t: self.eis[t][:, self.masks[t][2]]
        self.all = lambda t: self.eis[t]

        self.tr_attributes = lambda t: self.ews[t][self.masks[t][0]]
        self.va_attributes = lambda t: self.ews[t][self.masks[t][1]]
        self.te_attributes = lambda t: self.ews[t][self.masks[t][2]]
        self.all_attributes = lambda t: self.ews[t]

        # To match Euler models
        self.xs = self.x
        self.x_dim = self.x.size(1)

    def get_masked_edges(self, t, mask):
        if mask == self.TR:
            return self.tr(t)
        elif mask == self.VA:
            return self.va(t)
        elif mask == self.TE:
            return self.te(t)
        elif mask == self.ALL:
            return self.all(t)
        else:
            raise ArgumentError("Mask must be TData.TR, TData.VA, TData.TE, or TData.ALL")

    def ei_masked(self, mask, t):
        '''
        So method sig matches Euler models
        '''
        return self.get_masked_edges(t, mask)

    def ew_masked(self, *args):
        return None



def load_gc_data(dataset_path, num_nodes):
    print('in load gc 14')
    with open(dataset_path, 'rb') as f:
        dense_adj_list = pickle.load(f)
    eis = []
    ews_before = []
    splits = []

    for idx, adj in enumerate(dense_adj_list):
        eis.append(adj[0])
        ews_before.append(adj[1])
        # ews.append(adj.edge_attr)
        # adj = adj.edge_index
        splits.append(edge_tvt_split(adj[0]))

    ews = normalized(ews_before)
    data = TData(
        x=torch.eye(num_nodes),
        eis=eis,
        ews=ews,
        masks=splits,
        num_nodes=num_nodes,
        dynamic_feats=False,
        T=len(eis)
    )

    return data