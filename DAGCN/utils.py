from __future__ import print_function
import argparse
import sys

import networkx as nx
import numpy as np
import torch

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', type=int, default=-1, help='-1 for cpu, 0...n for GPU number')
cmd_opt.add_argument('-data', default='PTC', help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of node feature')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension of latent layers')
cmd_opt.add_argument('-out_dim', type=int, default=0, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=128, help='dimension of regression')
cmd_opt.add_argument('-max_lv', type=int, default=1, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-multi_h_emb_weight', type=int, default=1, help='multi_h_emb_weight')
cmd_opt.add_argument('-name', default='default.pk', help='10-cross-validation result')
cmd_opt.add_argument('-logDir', default='./log/default.txt', help='logfile directory')
cmd_opt.add_argument('-logDes', default='default', help='logfile description')
cmd_opt.add_argument('-max_k', type=int, default=3, help='k for capsules style')
cmd_opt.add_argument('-max_block', type=int, default=1, help='num of block layer')
cmd_opt.add_argument('-dropout', type=float, default=0.5)
cmd_opt.add_argument('-reg', type=int, default=0, help='regular term')

cmd_args, _ = cmd_opt.parse_known_args()


class S2VGraph(object):
    """将单个图转换为类"""
    def __init__(self, g, node_tags, label):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()


def load_data():
    """
    txt:
    第一行：图的数量 N
    块：
      当前图的节点数 n， 当前图的标签 l
      for i in range(n):
        第 i 个结点的标签 t , 以及邻居数量 m, 结点 i 的信息(邻居索引)
        for j in range(2, m):
             邻居[j]的索引
    """

    print('loading data ...')

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('./data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [int(w) for w in row]
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            # assert len(g.edges()) * 2 == n_edges
            assert len(g) == n
            g_list.append(S2VGraph(g, node_tags, l))
    for g in g_list:
        g.label = label_dict[g.label]

    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)

    print('# classes: %d' % cmd_args.num_class)
    print('# node features: %d' % cmd_args.feat_dim)

    train_idxes = np.loadtxt('./data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, cmd_args.fold),
                             dtype=np.int32).tolist()
    test_idxes = np.loadtxt('./data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, cmd_args.fold),
                            dtype=np.int32).tolist()

    return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]


def PrepareSparseMatrices(graph_list, is_directed=0):
    assert not is_directed
    total_num_nodes, total_num_edges = np.sum([[g.num_nodes, g.num_edges] for g in graph_list], axis=0)

    # 结点
    n2n_idxes = torch.LongTensor(2, total_num_edges * 2)
    n2n_idxes = torch.clip(n2n_idxes, 0, total_num_nodes-1)
    n2n_vals = torch.FloatTensor(total_num_edges * 2)

    # 边
    e2n_idxes = torch.LongTensor(2, total_num_edges * 2)
    e2n_idxes[0,:] = torch.clip(e2n_idxes[0,:], 0, total_num_nodes-1)
    e2n_idxes[1,:] = torch.clip(e2n_idxes[1,:], 0, total_num_edges * 2-1)
    e2n_vals = torch.FloatTensor(total_num_edges * 2)

    # 图
    subg_idxes = torch.LongTensor(2, total_num_nodes)
    subg_idxes[0,:] = torch.clip(subg_idxes[0,:], 0, len(graph_list)-1)
    subg_idxes[1,:] = torch.clip(subg_idxes[1,:], 0, total_num_nodes-1)

    subg_vals = torch.FloatTensor(total_num_nodes)

    # i, v, shape 构建稀疏邻接矩阵
    n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals,(total_num_nodes, total_num_nodes))
    e2n_sp = torch.sparse.FloatTensor(e2n_idxes, e2n_vals, (total_num_nodes, total_num_edges * 2))
    subg_sp =torch.sparse_coo_tensor(subg_idxes, subg_vals, (len(graph_list), total_num_nodes))

    return n2n_sp, e2n_sp, subg_sp
