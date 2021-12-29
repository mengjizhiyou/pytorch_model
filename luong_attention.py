import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch import Tensor
import torch.nn.functional as F


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, embedding_dim, hidden_dim):
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(embedding_dim, hidden_dim))
            embedding_dim = hidden_dim

    def forward(self, x, hx=None):
        # x: (N, E), B=N
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            if hx is None:
                h, c = layer(x)  # h:(N,H)
            else:
                h, c = layer(x, (hx[0][i], hx[1][i]))
            x = h
            h_list.append(h)
            c_list.append(c)
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        return x, (h_list, c_list)  # x:(N,H), h_list:(num_layer,N,H)


class Encoder(nn.Module):
    def __init__(self, num_layers, input_dim, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_dim * 100, embedding_dim)
        self.rnn = StackedLSTM(num_layers, embedding_dim, hidden_dim)

    def forward(self, x, hx=None):
        hs_bar_list = []
        x = self.embed(x.long())  # x:(N,S,E)
        for i in range(self.input_dim):
            hs_bar, hx = self.rnn(x[:, i, :], hx)  # (N,E),(N,H)-->(N,H),(num_layer, N,H)
            hs_bar_list.append(hs_bar)
        return torch.stack(hs_bar_list), hx  # (S,N,H),(num_layer, N,H)


class Decoder(nn.Module):
    def __init__(self, num_layers, input_dim, embedding_dim, hidden_dim, output_dim, attn_type, method='dot'):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_dim * 100, embedding_dim)
        self.rnn = StackedLSTM(num_layers, embedding_dim + hidden_dim, hidden_dim)
        self.Wc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tahn = nn.Tanh()
        if attn_type == 'global':
            self.attn = GlobalAttention(hidden_dim, method)
        else:
            self.attn = LocalAttention(input_dim, hidden_dim, method)
        self.Ws = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, enc_outputs, hx=None):
        x = self.embed(x.long())  # x:(N,S,E)
        output = []
        ht_tilde = enc_outputs[-1, :, :]  # (S,N,H)-->(N,H)
        for i in range(self.input_dim):
            x_cat = torch.cat([x[:, i, :], ht_tilde], dim=-1)  # x_cat: (N,E+H)
            ht, hx = self.rnn(x_cat, hx)  # ht:(N,H), hx:(num_layer,N,H)
            context, attn = self.attn(ht, enc_outputs)  # (N,1,H),(N,S)
            ht_tilde = self.tahn(self.Wc(torch.cat([context.squeeze(1), ht], dim=-1)))  # ht_tilde:(N,H)
            output.append(ht_tilde)
        return F.softmax(self.Ws(torch.stack(output)).transpose(0, 1),
                         dim=1)  # (S, N, output_dim) -> (N, S, output_dim)


class GlobalAttention(nn.Module):
    def __init__(self, hidden_dim, method='dot'):
        super(GlobalAttention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        if method == 'general':
            self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == 'concat':
            self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.bias = nn.Parameter(torch.FloatTensor(1, hidden_dim))
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs) -> Tuple[Tensor, Tensor]:
        # dec_hidden:(N,H) enc_outputs:(S,N,H)
        if self.method == 'general':
            score = torch.einsum('NH,SNH->NS', dec_hidden, self.Wa(enc_outputs))  # (N,S)
        elif self.method == 'dot':
            score = torch.einsum('NH,SNH->NS', dec_hidden, enc_outputs)  # (N,S)
        elif self.method == 'concat':
            score = self.v(enc_outputs * torch.tanh(self.Wa(dec_hidden + enc_outputs))).squeeze(-1)  # (N,S)

        attn = F.softmax(score, dim=-1)  # (N,S)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs.transpose(0, 1))  # (N,1,S)*(N,S,H)->(N,1,H)
        return context, attn  # (N,1,H),(N,S)


class LocalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, method='dot', D=3):
        super(LocalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, 1)
        self.v = nn.Linear(input_dim, 1)
        self.D = D
        self.attn = GlobalAttention(hidden_dim, method)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: (N,H)    enc_outputs:(S, N, H)
        S = enc_outputs.size(0)
        p = S * torch.sigmoid(self.v(torch.tanh(self.W(enc_outputs.transpose(0, 1)).squeeze(-1))))  # p:(N,1)
        sigma = self.D / 2
        context_list = []
        attn_list = []
        for i in range(p.size(0)):
            start, end = torch.clip(torch.tensor([p[i] - self.D, p[i] + self.D]), min=0, max=S)
            # context:(N,1,H), attn:(N,S=end-start)
            context, attn = self.attn(dec_hidden[i].unsqueeze(0), enc_outputs[start.int():end.int(), i, :].unsqueeze(1))
            exp = torch.exp(-(S - p[i]) ** 2 / (2 * sigma ** 2))
            context_list.append(context.squeeze(0) * exp)
            attn_list.append(attn.squeeze(0) * exp)
        return torch.stack(context_list), torch.stack(attn_list)


class neg_log_loss(nn.Module):
    """custom loss"""

    def __init__(self):
        super(neg_log_loss, self).__init__()

    def forward(self, output):
        return torch.sum(-torch.log(output))


class NMT(nn.Module):
    def __init__(self, num_layers, input_dim, embedding_dim, hidden_dim, output_dim, attn_type='local'):
        super(NMT, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = Encoder(num_layers, input_dim, embedding_dim, hidden_dim)
        self.decoder = Decoder(num_layers, input_dim, embedding_dim, hidden_dim, output_dim, attn_type)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, hx = self.encoder(enc_inputs)  # (S,N,H),(num_layer, N,H)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, hx)  # (N, S, output_dim)
        return dec_outputs


x = torch.arange(10).reshape(2, 5).float()
y = torch.arange(10, 20, 1).reshape(2, 5).float()
embedding_dim = 10
input_dim = x.size(1)
output_dim = 10
num_layers = 2
model = NMT(num_layers, input_dim, embedding_dim, hidden_dim=64, output_dim=output_dim)
# model = NMT(num_layers, input_dim, embedding_dim, hidden_dim=64, output_dim=output_dim, attn_type='global')
out = model(x, y)
print(out.shape)  # (2, 5, 10)
