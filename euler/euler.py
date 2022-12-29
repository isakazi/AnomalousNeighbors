import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, RGCNConv, GATv2Conv

from .drop_edge import DropEdge
from utils.loss_fns import full_adj_nll

# class RGCN(nn.Module):
#     def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
#         super(RGCN, self).__init__()

#         #self.lin = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
#         self.c1 = RGCNConv(feat_dim, hidden_dim, add_self_loops=True, num_relations=5000)
#         self.relu = nn.ReLU()
#         self.c2 = RGCNConv(hidden_dim, embed_dim, add_self_loops=True, num_relations=5000)
#         self.drop = nn.Dropout(0.25)
#         self.de = DropEdge(0.8)

#     def forward(self, x, ei, ew=None):
#         # print("Hello 3", x.shape, ei.shape)
#         ei = self.de(ei)
#         ei_new = []
#         ea = []
#         for i in range(len(ei)):
#             ei_new.append(ei[i][:2])
#             ea.append(ei[i][2])
#         x = self.c1(x, ei_new, edge_type=ea)
#         x = self.relu(x)
#         x = self.drop(x)

#         return self.c2(x, ei, edge_type=ea)

class GATE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
        super(GATE, self).__init__()

        # self.lin = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
        self.c1 = GATConv(feat_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GATConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.de = DropEdge(0.8)

    def forward(self, x, ei, ew=None):
        # print("Hello 4", x.shape, ei.shape)
        # print(x.type(), ei.type(), ew.type())
        ei = torch.vstack([ei, ew.transpose(0, 1)])
        ei = self.de(ei)
        ew = ei[2:].transpose(0, 1)  # .to(torch.float32)
        ei = ei[:2].to(torch.int64)
        # x = x.to(torch.float32)

        # print("Hello 5", x.type(), ei.type(), ew.type())
        # print("Hello 4", x.shape, ei.shape, ew.shape)
        # print(ew)
        # ew = ei[2].to(torch.float32)
        # ei = ei[:2]
        # for i in range(len(ei)):
        #     ei_new.append(ei[i][:2])
        #     ea.append(ei[i][2])
        # x = self.c1(x, ei, edge_attr=ew.transpose(0,1).to(torch.int64))
        # print("Hello 5", x.shape, ei.shape, ew.shape)
        x = self.c1(x=x, edge_index=ei, edge_attr=ew)
        x = self.relu(x)
        x = self.drop(x)

        return self.c2(x=x, edge_index=ei, edge_attr=ew)
        # return self.c2(x, ei, edge_attr=ew.transpose(0,1).to(torch.int64))


class GAE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
        super(GAE, self).__init__()

        # self.lin = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
        self.c1 = GCNConv(feat_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.de = DropEdge(0.8)

    #     def forward(self, x, ei, ew=None):
    #         print("Hello 2", x.shape, ei.shape
    #         ei = self.de(ei)
    #         x = self.c1(x, ei, edge_weight=ew)
    #         x = self.relu(x)
    #         x = self.drop(x)

    #         return self.c2(x, ei, edge_weight=ew)

    def forward(self, x, ei, ew=None):
        # print("Hello 3", x.shape, ei.shape)
        # print(ei)
        # print(ei)
        # print(ew)
        # ei = torch.vstack([ei, ew])
        ei, ew = self.de(ei, ew=ew)

        # ew = ei[2].to(torch.float32)
        # ei = ei[:2].to(torch.int64)

        # print(ei.shape, ew.shape)
        # print(ei)
        # print(ew)
        # ew = ei[1]
        # ei = ei[0]
        # ew = ei[2].to(torch.float32)
        # ei = ei[:2]
        # for i in range(len(ei)):
        #     ei_new.append(ei[i][:2])
        #     ew.append(ei[i][2])
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)

        return self.c2(x, ei, edge_weight=ew)


class Recurrent(nn.Module):
    def __init__(self, feat_dim, out_dim=16, hidden_dim=32, hidden_units=1, lstm=False):
        super(Recurrent, self).__init__()

        self.gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=hidden_units
        ) if not lstm else \
            nn.LSTM(
                feat_dim, hidden_dim, num_layers=hidden_units
            )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_dim, out_dim)

        self.out_dim = out_dim

    '''
    Expects (t, batch, feats) input
    Returns (t, batch, embed) embeddings of nodes at timesteps 0-t
    '''

    def forward(self, xs, h_0, include_h=False):
        xs = self.drop(xs)
        if type(h_0) != type(None):
            xs, h = self.gru(xs, h_0)
        else:
            xs, h = self.gru(xs)

        xs = self.drop(xs)
        if not include_h:
            return self.lin(xs)
        else:
            return self.lin(xs), h


class EulerGCN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1,
                 dynamic_feats=False, dense_loss=False,
                 use_predictor=False, use_w=True, lstm=False,
                 neg_weight=0.5):
        super(EulerGCN, self).__init__()

        self.weightless = not use_w
        self.kld_weight = 0
        self.dynamic_feats = dynamic_feats
        self.neg_weight = neg_weight
        self.cutoff = None
        self.z_dim = z_dim
        self.drop = nn.Dropout(0.05)

        print("using GAE")
        self.gcn = GAE(
            x_dim,
            hidden_dim=h_dim,
            embed_dim=h_dim if gru_hidden_units > 0 else z_dim
        )

        # print("using GATE")
        # self.gcn = GATE(
        #     x_dim,
        #     hidden_dim=h_dim,
        #     embed_dim=h_dim if gru_hidden_units > 0 else z_dim
        # )

        self.gru = Recurrent(
            h_dim, out_dim=z_dim,
            hidden_dim=h_dim,
            hidden_units=gru_hidden_units,
            lstm=lstm
        ) if gru_hidden_units > 0 else None

        self.use_predictor = use_predictor
        self.predictor = nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        ) if use_predictor else None

        self.sig = nn.Sigmoid()

        self.dense_loss = dense_loss
        msg = "dense" if self.dense_loss else 'sparse'
        print("Using %s loss" % msg)

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''

    def forward(self, xs, eis, mask_fn, ew_fn=None, start_idx=0,
                include_h=False, h_0=None):
        embeds = self.encode(xs, eis, mask_fn, ew_fn, start_idx)

        if type(self.gru) == type(None):
            if not include_h:
                return embeds
            else:
                return embeds, None
        else:
            return self.gru(torch.tanh(embeds), h_0, include_h=include_h)

    '''
    Split proceses in two to make it easier to combine embeddings with 
    different masks (ie allow train set to influence test set embeds)
    '''
    #     def encode(self, xs, eis, mask_fn, ew_fn=None, start_idx=0):
    #         embeds = []

    #         for i in range(len(eis)):
    #             ei = mask_fn(start_idx + i)
    #             ew = None if not ew_fn or self.weightless else ew_fn(start_idx + i)
    #             x = xs if not self.dynamic_feats else xs[start_idx + i]

    #             z = self.gcn(x,ei,ew)
    #             #print("ei: %s  ew: %s" % (str(ei.size()), str(ew.size())))
    #             embeds.append(z)

    #         return torch.stack(embeds)

    #     '''
    #     Iterates through list of xs, and eis passed in (if dynamic_feats is false
    #     assumes xs is a single 2d tensor that doesn't change through time)
    #     '''
    #     def forward(self, xs, eis, mask_fn, ew_fn=None, start_idx=0,
    #                 include_h=False, h_0=None):
    #         embeds = self.encode(xs, eis, mask_fn, ew_fn, start_idx)

    #         if type(self.gru) == type(None):
    #             if not include_h:
    #                 return embeds
    #             else:
    #                 return embeds, None
    #         else:
    #             return self.gru(torch.tanh(embeds), h_0, include_h=include_h)

    '''
    Split proceses in two to make it easier to combine embeddings with 
    different masks (ie allow train set to influence test set embeds)
    '''

    def encode(self, xs, eis, mask_fn, ew_fn=None, start_idx=0):
        embeds = []
        for i in range(len(eis)):
            ei = mask_fn(start_idx + i)
            ew = None if not ew_fn or self.weightless else ew_fn(start_idx + i)
            x = xs if not self.dynamic_feats else xs[start_idx + i]
            z = self.gcn(x, ei, ew)
            # print("ei: %s  ew: %s" % (str(ei.size()), str(ew.size())))
            embeds.append(z)

        return torch.stack(embeds)

    '''
    Inner product given edge list and embeddings at time t
    '''

    def decode(self, src, dst, z, as_probs=False):
        if self.use_predictor:
            return self.predictor(
                self.drop(z[src]) * self.drop(z[dst])
            )

        dot = (self.drop(z[src]) * self.drop(z[dst])).sum(dim=1)
        logits = self.sig(dot)

        if as_probs:
            return self.__logits_to_probs(logits)
        return logits

    '''
    Given confidence scores of true samples and false samples, return
    neg log likelihood 
    '''

    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores + EPS).mean()
        neg_loss = -torch.log(1 - f_scores + EPS).mean()

        return (1 - self.neg_weight) * pos_loss + self.neg_weight * neg_loss

    '''
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn  
    '''

    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)
        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            if not self.dense_loss:
                tot_loss += self.calc_loss(
                    self.decode(t_src, t_dst, z),
                    self.decode(f_src, f_dst, z)
                )
            else:
                tot_loss += full_adj_nll(ts[i], z)
        return tot_loss.true_divide(T)

    #     def loss_fn(self, ts, fs, zs):
    #         tot_loss = torch.zeros((1))
    #         T = len(ts)

    #         for i in range(T):
    #             # print(ts[i].shape)
    #             # print(fs[i].shape)
    #             # if len(ts[i]) == 3:
    #             #     t_src, t_dst, _ = ts[i]
    #             # else:
    #             #     t_src, t_dst = ts[i]
    #             # if len(fs[i]) == 3:
    #             #     f_src, f_dst, _ = fs[i]
    #             # else:
    #             #     f_src, f_dst = fs[i]
    #             # t_src, t_dst, _ = ts[i]
    #             f_src, f_dst = fs[i]
    #             z = zs[i]

    #             if not self.dense_loss:
    #                 tot_loss += self.calc_loss(
    #                     self.decode(t_src, t_dst, z),
    #                     self.decode(f_src, f_dst, z)
    #                 )
    #             else:
    #                 tot_loss += full_adj_nll(ts[i], z)

    #         return tot_loss.true_divide(T)

    '''
    Get scores for true/false embeddings to find ROC/AP scores.
    Essentially the same as loss_fn but with no NLL 

    Returns logits unless as_probs is True
    '''

    def score_fn(self, ts, fs, zs, as_probs=False):
        tscores = []
        fscores = []

        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            tscores.append(self.decode(t_src, t_dst, z))
            fscores.append(self.decode(f_src, f_dst, z))

        tscores = torch.cat(tscores, dim=0)
        fscores = torch.cat(fscores, dim=0)

        if as_probs:
            tscores = self.__logits_to_probs(tscores)
            fscores = self.__logits_to_probs(fscores)

        return tscores, fscores

    #     def score_fn(self, ts, fs, zs, as_probs=False):
    #         tscores = []
    #         fscores = []

    #         T = len(ts)

    #         for i in range(T):
    #             if len(ts[i]) == 3:
    #                 t_src, t_dst, _ = ts[i]
    #             else:
    #                 t_src, t_dst = ts[i]
    #             if len(fs[i]) == 3:
    #                 f_src, f_dst, _ = fs[i]
    #             else:
    #                 f_src, f_dst = fs[i]
    #             z = zs[i]

    #             tscores.append(self.decode(t_src, t_dst, z))
    #             fscores.append(self.decode(f_src, f_dst, z))

    #         tscores = torch.cat(tscores, dim=0)
    #         fscores = torch.cat(fscores, dim=0)

    #         if as_probs:
    #             tscores=self.__logits_to_probs(tscores)
    #             fscores=self.__logits_to_probs(fscores)

    #         return tscores, fscores

    '''
    Converts from log odds (what the encode method outputs) to probabilities
    '''

    def __logits_to_probs(self, logits):
        odds = torch.exp(logits)
        probs = odds.true_divide(1 + odds)
        return probs