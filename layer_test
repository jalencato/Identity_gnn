import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np
import networkx as nx
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import CoraGraphDataset
from torch.nn import Parameter
from dgl.utils import expand_as_pair

# gcn_msg = fn.copy_u(u='h', out='m')
# gcn_reduce = fn.sum(msg='m', out='h')
def gcn_msg(edge):
    msg = edge.src['h']
    return {'m': msg}


def gcn_reduce(msg):
    accum = th.sum(msg.mailbox['m'], dim=1)
    return {'h': accum}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.weight = Parameter(th.Tensor(in_feats, out_feats))
        self.weight_id = Parameter(th.Tensor(in_feats, out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1)
        nn.init.xavier_uniform_(self.weight_id, gain=1)

    def forward(self, g, feature, id=th.tensor([0, 1])):
        # x_id = torch.index_select(x, dim=0, index=id)
        # x_id = torch.matmul(x_id, self.weight_id)
        # x = torch.matmul(x, self.weight)
        # x.index_add_(0, id, x_id)

        # feature_id = th.index_select(feature, dim=1, index=id)
        # id = th.tensor([0, 1])
        feature_id = th.index_select(feature, dim=0, index=id)
        feature_id = th.matmul(feature_id, self.weight_id)
        feature = th.mm(feature, self.weight)
        feature.index_add_(0, id, feature_id)
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        feature = g.ndata.pop('h')

        return feature
        # self.g.ndata['h'] = h
        # self.g.update_all(fn.copy_src(src='h', out='m'),
        #                   fn.sum(msg='m', out='h'))
        # h = self.g.ndata.pop('h')
        # # normalization by square root of dst degree
        # h = h * self.g.ndata['norm']
        # return self.linear(h)

class Net(nn.Module):
    def __init__(self, in_feat=None, n_classes=None, n_hidden=16,
                 dropout=0.5,
                 activation=F.relu):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        #add layers
        self.layers.append(GCNLayer(in_feat, n_hidden))
        self.layers.append(GCNLayer(n_hidden, n_hidden))

    def forward(self, graph, features):
        h = self.dropout(features)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    return g, features, labels, train_mask, test_mask, dataset.num_classes


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


if __name__ == '__main__':
    g, features, labels, train_mask, test_mask, n_classes = load_cora_data()
    g = dgl.add_self_loop(g)
    net = Net(features.shape[1], n_classes)
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
    print(net)

    for epoch in range(50):
        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f}".format(
            epoch, loss.item(), acc))

    acc = evaluate(net, g, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
