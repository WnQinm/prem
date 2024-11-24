import pickle

import torch
import torch.nn as nn
import dgl.function as fn
import os
import sys

from .kan import KAN


class Dataloader:
    def __init__(self, g, features, k, dataset_name = None):
        self.k = k
        self.g = g
        self.label_zeros = torch.zeros(1, g.number_of_nodes()).to(features.device)
        self.label_ones = torch.ones(1, g.number_of_nodes()).to(features.device)

        self.en = features.detach()
        if dataset_name is not None and os.path.isfile(f"./cache/{dataset_name}.pickle"):
            # print(f"Load precomputed graph emb from ./cache/{dataset_name}.pickle")
            with open(f"./cache/{dataset_name}.pickle", "rb") as fp:
                precomputed = pickle.load(fp)
                self.weight = precomputed["weight"].to(features.device)
                self.features_weighted = precomputed["features_weighted"].to(features.device)
                self.eg = precomputed["eg"].to(features.device)

        else:
            # print("Preprocessing: Aggregrate neighbour embeddings")
            self.weight = get_diag(self.g, self.k)
            aggregated = aggregation(self.g, features, self.k)
            self.features_weighted = (features.swapaxes(1, 0) * self.weight).swapaxes(1, 0).detach()
            self.eg = (aggregated - self.features_weighted).detach()
            if dataset_name is not None:
                # print(f"Save graph emb to ./cache/{dataset_name}.pickle")
                if not os.path.isdir("./cache"):
                    os.makedirs("./cache")
                with open(f"./cache/{dataset_name}.pickle", "wb") as fp:
                    pickle.dump({
                        "weight": self.weight.to("cpu"),
                        "features_weighted": self.features_weighted.to("cpu"),
                        "eg": self.eg.to("cpu")
                    }, fp)

    def get_data(self):
        en_p = self.en
        eg_p = self.eg
        perm = torch.randperm(en_p.shape[0])
        en_n = en_p[perm]
        eg_aug = eg_p[perm]
        return en_p, en_n, eg_p, eg_aug


def aggregation(graph, feat, k):
    with graph.local_scope():
        # compute normalization
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(feat.device).unsqueeze(1)
        # compute (D^-1 A^k D^-1)^k X
        for _ in range(k):
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            feat = feat * norm
        return feat


def get_diag(graph, k):
    aggregated_matrix = aggregation(
        graph,
        torch.eye(graph.num_nodes(), graph.num_nodes()).to(graph.device),
        k
    )
    return torch.diag(aggregated_matrix)


class Model(nn.Module):
    def __init__(self, g, n_in, n_hidden, k):
        super(Model, self).__init__()
        self.g = g
        self.k = k
        # self.fc_g = KAN(width=[n_in, n_hidden, n_hidden], grid=5, k=3, auto_save=False)
        # self.fc_n = KAN(width=[n_in, n_hidden, n_hidden], grid=5, k=3, auto_save=False)
        self.fc_g = nn.Linear(n_in, n_hidden)
        self.fc_n = nn.Linear(n_in, n_hidden)

    def forward(self, target_features, neighbour_features):
        score = torch.nn.functional.cosine_similarity(self.fc_n(target_features.detach()), self.fc_g(neighbour_features.detach()))
        return -1 * score.unsqueeze(0)

    def before_train(self, lamb=0):
        self.old_save_act, self.old_symbolic_enabled = zip(self.fc_g.disable_symbolic_in_fit(lamb), self.fc_n.disable_symbolic_in_fit(lamb))

    def after_train(self):
        self.fc_g.save_act, self.fc_n.save_act = self.old_save_act
        self.fc_g.symbolic_enabled, self.fc_n.symbolic_enabled = self.old_symbolic_enabled

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.fc_g.saveckpt(path+"/fc_g")
        self.fc_n.saveckpt(path+"/fc_n")

    def load(self, path):
        self.fc_g = KAN.loadckpt(path+"/fc_g")
        self.fc_n = KAN.loadckpt(path+"/fc_n")

    @torch.no_grad()
    def update_grid(self, target_features, neighbour_features):
        self.fc_g.update_grid(target_features)
        self.fc_n.update_grid(neighbour_features)
