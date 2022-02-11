import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from MLP import MLP
from dagcn import DAGCN
from utils import cmd_args, load_data


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        model = DAGCN

        self.gnn = model(latent_dim=cmd_args.latent_dim,
                         output_dim=cmd_args.out_dim,
                         num_node_feats=cmd_args.feat_dim,
                         num_edge_feats=0,
                         multi_h_emb_weight=cmd_args.multi_h_emb_weight,
                         max_k=cmd_args.max_k,
                         dropout=cmd_args.dropout,
                         max_block=cmd_args.max_block,
                         reg=cmd_args.reg)

        self.mlp = MLP(input_size=cmd_args.latent_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        concat_feat = []
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            concat_feat += batch_graph[i].node_tags

        concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
        node_feat = torch.zeros(n_nodes, cmd_args.feat_dim)
        node_feat.scatter_(1, concat_feat, 1)

        if cmd_args.mode >= 0:
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed, reg = self.gnn(batch_graph, node_feat, None)
        return self.mlp(embed, labels, reg)


def loop_dataset(g_list, classifier, sample_idxes, train=True, optimizer=None, bsize=cmd_args.batch_size):
    if train:
        classifier.train()
    else:
        classifier.eval()
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch', leave=False)

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        _, loss, acc = classifier(batch_graph)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loss = loss.data[0]
        loss = loss.item()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss


if __name__ == '__main__':
    if cmd_args.mode >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cmd_args.mode)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    classifier = Classifier()

    if cmd_args.mode >= 0:
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)  # Note: any better optimizer?

    train_idxes = list(range(len(train_graphs)))

    best_valid_acc = test_acc = 0
    old_epoch_records = []

    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)

        train_loss, train_acc = loop_dataset(train_graphs, classifier, train_idxes, train=True, optimizer=optimizer)
        print('maverage training of epoch %d: loss %.5f acc %.5f' % (epoch, train_loss, train_acc))

        test_loss, test_acc = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))), train=False, )
        print('maverage test of epoch %d: loss %.5f acc %.5f' % (epoch, test_loss, test_acc))


        # if (epoch+1) % 5 == 0:
        #     for name, param in classifier.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.data)
