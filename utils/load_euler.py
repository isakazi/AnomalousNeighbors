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

        # Getter methods so I don't have to write this every time
        #         self.tr = lambda t : self.eis[t].edge_index[:, self.masks[t][0]]
        #         self.va = lambda t : self.eis[t].edge_index[:, self.masks[t][1]]
        #         self.te = lambda t : self.eis[t].edge_index[:, self.masks[t][2]]
        #         self.all = lambda t : self.eis[t].edge_index

        #         self.tr_attributes = lambda t : self.eis[t].edge_attr[self.masks[t][0]]
        #         self.va_attributes = lambda t : self.eis[t].edge_attr[self.masks[t][1]]
        #         self.te_attributes = lambda t : self.eis[t].edge_attr[self.masks[t][2]]
        #         self.all_attributes = lambda t : self.eis[t].edge_attr

        self.tr = lambda t: self.eis[t][:, self.masks[t][0]]
        self.va = lambda t: self.eis[t][:, self.masks[t][1]]
        self.te = lambda t: self.eis[t][:, self.masks[t][2]]
        self.all = lambda t: self.eis[t]

        self.tr_attributes = lambda t: self.ews[t][self.masks[t][0]]
        self.va_attributes = lambda t: self.ews[t][self.masks[t][1]]
        self.te_attributes = lambda t: self.ews[t][self.masks[t][2]]
        self.all_attributes = lambda t: self.ews[t]

        #         self.tr_attributes = lambda t : self.eis[t][1][self.masks[t][0]]
        #         self.va_attributes = lambda t : self.eis[t][1][self.masks[t][1]]
        #         self.te_attributes = lambda t : self.eis[t][1][self.masks[t][2]]
        #         self.all_attributes = lambda t : self.eis[t][1]
        # self.tr_attributes = lambda t : torch.transpose(torch.transpose(self.eas[t],0,1)[:, self.masks[t][0]], 0,1)
        # self.va_attributes = lambda t : torch.transpose(torch.transpose(self.eas[t],0,1)[:, self.masks[t][1]], 0,1)
        # self.te_attributes = lambda t : torch.transpose(torch.transpose(self.eas[t],0,1)[:, self.masks[t][2]], 0,1)
        # self.all_attributes = lambda t : self.eas[t]

        # self.tr = lambda t : self.eis[t][:, self.masks[t][0]]
        # self.va = lambda t : self.eis[t][:, self.masks[t][1]]
        # self.te = lambda t : self.eis[t][:, self.masks[t][2]]
        # self.all = lambda t : self.eis[t]

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
        '''
        VGRNN datasets don't have weighted edges
        '''
        return None


'''
For loading datasets from the VRGNN repo (none have features)
'''


def load_vgrnn(dataset):
    datasets = ['fb', 'dblp', 'enron10', 'gc']
    assert dataset in datasets, \
        "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

    adj = os.path.join('/home/jupyter/Euler/benchmarks/data', dataset, 'adj_orig_dense_list.pickle')
    with open(adj, 'rb') as f:
        fbytes = f.read()

    dense_adj_list = pickle.loads(fbytes, encoding='bytes')
    num_nodes = max([dense_adj_list[i].size(0) for i in range(len(dense_adj_list))])

    eis = []
    splits = []

    for idx, adj in enumerate(dense_adj_list):
        # Remove self loops
        for i in range(adj.size(0)):
            adj[i, i] = 0

        ei = dense_to_sparse(adj)[0]
        ei = to_undirected(ei)
        eis.append(ei)

        if idx == len(dense_adj_list) - 1:
            print(f'doing stupid things {idx}')
            splits.append(edge_tvt_split(ei, False))
        else:
            splits.append(edge_tvt_split(ei))

    data = TData(
        x=torch.eye(num_nodes),
        eis=eis,
        masks=splits,
        num_nodes=num_nodes,
        dynamic_feats=False,
        T=len(eis)
    )

    return data


# def load_gc_data(dataset):
#     print('in load gc 9')
#     datasets = ['gc']
#     assert dataset in datasets, \
#         "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

#     adj = os.path.join('/home/jupyter/Euler/benchmarks/data', dataset, 'adj_orig_dense_list.pickle')
#     dense_adj_list = []
#     with open(adj, 'rb') as f:
#         dense_adj_list = pickle.load(f)
#     # num_nodes = max([torch.unique(dense_adj_list[i]).size(0) for i in range(len(dense_adj_list))])
#     num_nodes =  dense_adj_list[0].num_nodes #26008 #7827 # 4276 #58387 #4276 #2442
#     # num_nodes = max([dense_adj_list[i].size(0) for i in range(len(dense_adj_list))])
#     eis = []
#     eas = []
#     splits = []

#     for idx,adj in enumerate(dense_adj_list):
#         # Remove self loops
#         ea = adj.edge_attr
#         adj = adj.edge_index
#         eis.append(adj)
#         eas.append(ea)
#         if idx == len(dense_adj_list)-1:
#             # print(f'doing stupid things {idx}')
#             splits.append(edge_tvt_split(adj))
#         else:
#             splits.append(edge_tvt_split(adj))

#     data = TData(
#         x=torch.eye(num_nodes),
#         eis=eis,
#         eas = eas,
#         masks=splits,
#         num_nodes=num_nodes,
#         dynamic_feats=False,
#         T=len(eis)
#     )

#     return data
# def load_gc_data(dataset):
#     print('in load gc 7')
#     datasets = ['gc']
#     assert dataset in datasets, \
#         "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

#     adj = os.path.join('/home/jupyter/Euler/benchmarks/data', dataset, 'adj_orig_dense_list.pickle')
#     dense_adj_list = []
#     with open(adj, 'rb') as f:
#         dense_adj_list = pickle.load(f)
#     # num_nodes = max([torch.unique(dense_adj_list[i]).size(0) for i in range(len(dense_adj_list))])
#     num_nodes = 59662 # 17160 #59662 # 26008 #7827 # 4276 #58387 #4276 #2442
#     # num_nodes = max([dense_adj_list[i].size(0) for i in range(len(dense_adj_list))])
#     eis = []
#     splits = []

#     for idx,adj in enumerate(dense_adj_list):
#         # Remove self loops
#         eis.append(adj)

#         if idx == len(dense_adj_list)-1:
#             # print(f'doing stupid things {idx}')
#             splits.append(edge_tvt_split(adj))
#         else:
#             splits.append(edge_tvt_split(adj))

#     data = TData(
#         x=torch.eye(num_nodes),
#         eis=eis,
#         masks=splits,
#         num_nodes=num_nodes,
#         dynamic_feats=False,
#         T=len(eis)
#     )

#     return data


# def load_gc_data(dataset):
#     print('in load gc 11')
#     datasets = ['gc']
#     assert dataset in datasets, \
#         "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

#     adj = os.path.join('/home/jupyter/Euler/benchmarks/data', dataset, 'adj_orig_dense_list.pickle')
#     dense_adj_list = []
#     with open(adj, 'rb') as f:
#         dense_adj_list = pickle.load(f)
#     # num_nodes = max([torch.unique(dense_adj_list[i]).size(0) for i in range(len(dense_adj_list))])
#     num_nodes = 1323 #59662 # 17160 #59662 # 26008 #7827 # 4276 #58387 #4276 #2442
#     # num_nodes = max([dense_adj_list[i].size(0) for i in range(len(dense_adj_list))])
#     eis = []
#     splits = []

#     for idx,adj in enumerate(dense_adj_list):
#         # Remove self loops
#         eis.append(adj)

#         if idx == len(dense_adj_list)-1:
#             # print(f'doing stupid things {idx}')
#             splits.append(edge_tvt_split(adj))
#         else:
#             splits.append(edge_tvt_split(adj))

#     data = TData(
#         x=torch.eye(num_nodes),
#         eis=eis,
#         masks=splits,
#         num_nodes=num_nodes,
#         dynamic_feats=False,
#         T=len(eis)
#     )

#     return data


# def load_gc_data(dataset):
#     print('in load gc 13')
#     datasets = ['gc']
#     assert dataset in datasets, \
#         "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

#     adj = os.path.join('/home/jupyter/Euler/benchmarks/data', dataset, 'adj_orig_dense_list.pickle')
#     dense_adj_list = []
#     with open(adj, 'rb') as f:
#         dense_adj_list = pickle.load(f)
#     # num_nodes = max([torch.unique(dense_adj_list[i]).size(0) for i in range(len(dense_adj_list))])
#     num_nodes = 22234 # 24157 #5124 # 6177 # 12091 #13660 #8878#79772 #1323 #59662 # 17160 #59662 # 26008 #7827 # 4276 #58387 #4276 #2442
#     # num_nodes = max([dense_adj_list[i].size(0) for i in range(len(dense_adj_list))])
#     eis = []
#     ews = []
#     splits = []

#     for idx,adj in enumerate(dense_adj_list):
#         # Remove self loops
#         eis.append(adj)
#         # ews.append(adj.edge_attr)
#         # adj = adj.edge_index
#         if idx == len(dense_adj_list)-1:
#             # print(f'doing stupid things {idx}')
#             splits.append(edge_tvt_split(adj))
#         else:
#             splits.append(edge_tvt_split(adj))

#     data = TData(
#         x=torch.eye(num_nodes),
#         eis=eis,
#         ews=ews,
#         masks=splits,
#         num_nodes=num_nodes,
#         dynamic_feats=False,
#         T=len(eis)
#     )

#     return data


####### WITH WEIGHTS ######
def load_gc_data(dataset, num_nodes=22234):
    # print('in load gc 14')
    datasets = ['gc']
    assert dataset in datasets, \
        "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

    adj = os.path.join('/home/jupyter/Euler/benchmarks/data', dataset, 'adj_orig_dense_list.pickle')
    dense_adj_list = []
    with open(adj, 'rb') as f:
        dense_adj_list = pickle.load(f)
    # num_nodes = max([torch.unique(dense_adj_list[i]).size(0) for i in range(len(dense_adj_list))])
    num_nodes = 22234  # 24157 #5124 # 6177 # 12091 #13660 #8878#79772 #1323 #59662 # 17160 #59662 # 26008 #7827 # 4276 #58387 #4276 #2442
    # num_nodes = max([dense_adj_list[i].size(0) for i in range(len(dense_adj_list))])
    eis = []
    ews = []
    ews_before = []
    splits = []

    for idx, adj in enumerate(dense_adj_list):
        # Remove self loops
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