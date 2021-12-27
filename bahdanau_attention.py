# 《NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE》
# 通过学习对齐翻译的神经机器翻译

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    """创建编码器"""

    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim * 20, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x, hx=None):
        x = self.embedding(x.long())
        if hx is None:
            output, (h, c) = self.lstm(x)  # output:(batch_size, seq_len, 2*hidden_dim)
        else:
            output, (h, c) = self.lstm(x, hx)  # h:(2, batch_size, hidden_dim//2)
        return output, (h, c)


class Decoder(nn.Module):
    """创建解码器"""

    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim * 20, embedding_dim)
        # self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        """update gates z_t"""
        self.Wz = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.Uz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Cz = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        """update gates r_t"""
        self.Wr = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.Ur = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Cr = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        """update state"""
        self.W = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False)
        self.U = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.C = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)

    def update_state(self, y_t_minus_1, s_t_minus_1, c_t):
        """
        y_t_minus_1: [batch_size, 1, embed_size]
        s_t_minus_1: (N,1,H)
        c_t: (N,1,2*H)
        """
        z_t = torch.sigmoid(self.Wz(y_t_minus_1) + self.Uz(s_t_minus_1) + self.Cz(c_t))
        r_t = torch.sigmoid(self.Wr(y_t_minus_1) + self.Ur(s_t_minus_1) + self.Cr(c_t))
        s_tilde = torch.tanh(self.W(y_t_minus_1) + self.U(r_t * s_t_minus_1) + self.C(c_t))
        s_t = (1 - z_t) * s_t_minus_1 + z_t * s_tilde
        return s_t

    def forward(self, s_t_minus_1, y_t_minus_1, c_t):
        y_t_minus_1 = self.embedding(y_t_minus_1.long())  # (N,1,E)
        s_t = self.update_state(y_t_minus_1, s_t_minus_1, c_t)  # (N,1,H)
        return s_t


class AdditiveAttention(nn.Module):
    """构建对齐模型,计算attention"""

    def __init__(self, hidden_dim) -> None:
        super(AdditiveAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)  # (N,H)->(N,H)
        self.U = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # (N,S,2H)->(N,S,H)
        self.v = nn.Linear(hidden_dim, 1)  #
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.tanh = nn.Tanh()

    def forward(self, enc_outputs: Tensor, dec_hidden: Tensor) -> Tuple[Tensor, Tensor]:
        # (N,S,H)->(N,S,1)->(N,S)
        score = self.v(self.tanh(self.W(dec_hidden) + self.U(enc_outputs) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)  # (B,S)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs)  # (N,1,S)*(N,S,2*H)->(N,1,2*H)
        return context, attn


class NMT(nn.Module):
    """将3个整合到一起"""

    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(NMT, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(input_dim, embedding_dim, hidden_dim)
        self.decoder = Decoder(input_dim, embedding_dim, hidden_dim)
        self.attention = AdditiveAttention(hidden_dim)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, (enc_h, enc_c) = self.encoder(enc_inputs)
        dec_hidden = enc_outputs[:, -1, :self.hidden_dim].unsqueeze(1)  # (N,1,H)
        for i in range(dec_inputs.size(1)):
            dec_input = dec_inputs[:, i].unsqueeze(1)  # (N,1)
            context, attn = self.attention(enc_outputs, dec_hidden)
            if i == 0:
                dec_hidden = self.decoder(dec_hidden, dec_input, context)
                out = dec_hidden.cpu().detach().numpy()
            else:
                dec_hidden = self.decoder(dec_hidden, dec_input, context)  # (N, 1, hidden_dim)
                out = np.hstack((out, dec_hidden.cpu().detach().numpy()))
        return out.reshape(-1, dec_inputs.size(1), self.hidden_dim)


x = torch.arange(10).reshape(1, 10).float()
y = torch.arange(10, 15, 1).reshape(1, 5).float()
embedding_dim = 10
input_dim = x.size(1)
model = NMT(input_dim, embedding_dim, hidden_dim=64)
out = model(x, y)
print(out.shape)  # (1, 5, 64)
