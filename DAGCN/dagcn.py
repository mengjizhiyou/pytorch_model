from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from utils import PrepareSparseMatrices


class DAGCN(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats,
                 multi_h_emb_weight, max_k=3, max_block=3, dropout=0.3, reg=1):
        """
        num_node_feats:结点特征维度
        num_edge_feats:边特征维度
        max_k: hop 的个数
        max_block: AGC layer 的层数
        reg: regular term 正则化项
        """
        print('Dual Attentional Graph Convolution')
        super(DAGCN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.dropout = dropout
        self.reg = reg
        self.multi_h_emb_weight = multi_h_emb_weight

        self.max_k = max_k
        self.max_block = max_block

        #  节点特征转换
        self.node = nn.Sequential(nn.Linear(num_node_feats, latent_dim), nn.BatchNorm1d(latent_dim))

        # 边特征转换
        if num_edge_feats > 0:
            self.edge = nn.Sequential(nn.Linear(num_edge_feats, latent_dim), nn.BatchNorm1d(latent_dim))

        if output_dim > 0:
            self.AGC_out = nn.Sequential(nn.Linear(latent_dim, output_dim), nn.BatchNorm1d(output_dim),
                                         nn.ReLU(inplace=True))

        # Initial weights,
        self.conv_params = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for _ in range(max_k))
        self.bn2 = nn.ModuleList(nn.BatchNorm1d(latent_dim) for _ in range(max_k))

        # Node level attention // hop attention
        self.k_weight = Parameter(torch.Tensor(self.max_k * latent_dim, latent_dim))
        self.bn3 = nn.BatchNorm1d(latent_dim)

        in_dim = output_dim if output_dim > 0 else latent_dim
        # Graph level attention // AGC layer attention
        self.graph_attn = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.Tanh(), nn.BatchNorm1d(latent_dim),
                                        nn.Linear(latent_dim, multi_h_emb_weight), nn.BatchNorm1d(multi_h_emb_weight))
        self.reset_params()

    def reset_params(self):
        # 初始化可学习的参数
        for m in self.modules():
            if isinstance(m, nn.ParameterList):
                for p in m: nn.init.kaiming_normal_(p.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        for name, p in self.named_parameters():
            if not '.' in name:  # top-level parameters
                nn.init.kaiming_normal_(p.data)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]

        dv = node_feat.device

        n2n_sp, e2n_sp, subg_sp = PrepareSparseMatrices(graph_list)
        n2n_sp, e2n_sp, subg_sp = n2n_sp.to(dv), e2n_sp.to(dv), subg_sp.to(dv)

        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)

        h = self.AGC(node_feat, edge_feat, n2n_sp, e2n_sp)

        h = self.APooling(h, subg_sp, graph_sizes)

        reg_term = self.get_reg(self.reg)

        return h, (reg_term / subg_sp.size(0))

    def AGC(self, node_feat, edge_feat, n2n_sp, e2n_sp):
        node_message = self.node(node_feat)

        if edge_feat is not None:
            edge_message = self.edge(edge_feat)
            node_message += torch.spmm(e2n_sp, edge_message)

        input = F.relu(node_message)

        block = 0
        cur_message_layer = input
        A = n2n_sp

        while block < self.max_block:
            if block == 0:
                block_input = cur_message_layer
            else:
                block_input = cur_message_layer + input
            h = self.multi_hop(block_input, A)
            h = F.relu(h)
            cur_message_layer = h
            block += 1

        if self.output_dim > 0:
            node_emb = self.AGC_out(cur_message_layer)
        else:
            node_emb = cur_message_layer

        return node_emb

    def multi_hop(self, cur_message_layer, A):
        step = 0
        input_x = cur_message_layer
        n, m = cur_message_layer.shape
        result = torch.zeros((n, m * self.max_k)).to(A.device)
        while step < self.max_k:  # 这里的A是非归一化邻接矩阵
            n2npool = torch.spmm(A, input_x) + cur_message_layer  # Y = (A + I) * X
            input_x = self.conv_params[step](n2npool)  # Y = Y * W
            input_x = self.bn2[step](input_x)
            result[:, (step * self.latent_dim):(step + 1) * self.latent_dim] = input_x[:, :]
            step += 1
        return self.bn3(torch.matmul(result, self.k_weight).view(n, -1))  # (N, latent_dim)

    def APooling(self, node_emb, subg_sp, graph_sizes):
        attn = self.graph_attn(node_emb)
        graph_emb = torch.zeros(len(graph_sizes), self.multi_h_emb_weight, self.latent_dim)
        graph_emb = graph_emb.to(node_emb.device)
        graph_emb = Variable(graph_emb)

        accum_count = 0
        for i in range(subg_sp.size(0)):
            alpha = attn[accum_count: accum_count + graph_sizes[i]]
            alpha = F.softmax(alpha, dim=-1)

            alpha = F.dropout(alpha, self.dropout)
            alpha = alpha.t()

            input_before = node_emb[accum_count: accum_count + graph_sizes[i]]
            emb_g = torch.matmul(alpha, input_before)

            graph_emb[i] = emb_g
            accum_count += graph_sizes[i]

        y_potential = graph_emb.view(len(graph_sizes), -1)
        return F.relu(y_potential)

    def get_reg(self, r=None):
        reg = 0
        for p in self.parameters():
            if p.dim() > 1:
                if r == 1:
                    reg += abs(p).sum()
                elif r == 2:
                    reg += p.pow(2).sum()
        return reg
