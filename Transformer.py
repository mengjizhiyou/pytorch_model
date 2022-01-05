import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import numpy as np
import torch.nn.functional as F
import math


class ScaledDotAttention(nn.Module):
    def __init__(self, d_k):
        """d_k: attention 的维度"""
        super(ScaledDotAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # q:nqhd->nhqd, k:nkhd->nhkd->nhdk  nhqd*nhdk->nhqk
        score = torch.einsum("nqhd,nkhd->nhqk", [q, k]) / np.sqrt(self.d_k)

        if mask is not None:
            # 将mask为0的值，填充为负无穷，则在softmax时权重为0（被屏蔽的值不考虑）
            score.masked_fill_(mask == 0, -float('Inf'))

        attn = F.softmax(score, -1)  # nhqk

        # score:nhqk   v:nkhd->nhkd   nhqk*nhkd=nhqd=nqhd
        context = torch.einsum("nhqk,nkhd->nqhd", [attn, v])  # nqhd
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        """
        d_model: q/k/v 的输入维度
        num_heads: attention的个数
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 等于embedding_dim
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model % num_heads should be zero"
        self.d_k = d_model // num_heads
        self.scaled_dot_attn = ScaledDotAttention(self.d_k)
        self.W_Q = nn.Linear(self.d_k, self.d_k, bias=False)
        self.W_K = nn.Linear(self.d_k, self.d_k, bias=False)
        self.W_V = nn.Linear(self.d_k, self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        """
        query:(batch, q_len, d_model):来自前一个decoder层；来自输入；来自输出
        key:(batch, k_len, d_model):来自编码器的输出；来自输入；来自输出
        value:(batch, v_len, d_model):来自编码器的输出；来自输入；来自输出
        """

        N = value.size(0)  # batch_size

        # 转化成8个注意,平行运行

        query = query.view(N, -1, self.num_heads, self.d_k)  # N*q_len*h*d
        key = key.view(N, -1, self.num_heads, self.d_k)  # N*k_len*h*d
        value = value.view(N, -1, self.num_heads, self.d_k)  # N*v_len*h*d ;  k_len=v_len

        query = self.W_Q(query)
        key = self.W_K(key)
        value = self.W_V(value)

        context, attn = self.scaled_dot_attn(query, key, value, mask)  # nhqk
        context = self.W_O(context.reshape(N, query.size(1), self.num_heads * self.d_k))  # N*q_len*(h*d=d_model)

        return context, attn


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        for pos in range(max_len):
            for i in range(d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads=8, dropout=0.1):
        """
        dropout 应用于每一个子层
        """
        super(EncoderBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model))  # 前馈网络：两个线性+1个激活

    def forward(self, query, key, value, mask):
        context, _ = self.attn(query, key, value, mask)
        # 跳跃连接
        x = self.dropout(self.norm1(context + query))
        forward = self.FFN(x)
        out = self.dropout(self.norm2(x + forward))
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model=512, d_ff=2048, num_layers=1, num_heads=8, dropout=0.1, max_len=500):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = nn.Dropout()
        self.word_embedding = nn.Embedding(input_dim * 200, d_model)
        self.position_embedding = PositionEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        out = self.dropout(self.word_embedding(x.long()) * math.sqrt(self.d_model) + self.position_embedding(x))

        # 在编码器中，Q、K、V是一样的，但在解码器中会变化
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads=8, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.block = EncoderBlock(d_model, d_ff, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, tgt_mask):
        context, _ = self.attn(x, x, x, tgt_mask)
        query = self.dropout(self.norm(context + x))
        out = self.block(query, key, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, input_dim, d_model=512, d_ff=2048, num_layers=1, num_heads=8, dropout=0.1, max_len=500):
        """这里的input_dim表示目标语句的长短，可以源词的输入长度不同"""
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(input_dim * 200, d_model)
        self.position_embedding = PositionEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])
        self.W = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_outputs, src_mask, tgt_mask):
        x = self.dropout(self.word_embedding(x.long()) * math.sqrt(self.d_model) + self.position_embedding(x))

        # 在编码器中，Q、K、V是一样的，但在解码器中会变化
        for layer in self.layers:
            x = layer(x, enc_outputs, enc_outputs, src_mask, tgt_mask)
        out = self.W(x)
        return out


class Transformer(nn.Module):
    def __init__(self, src_input_dim, tgt_input_dim,
                 src_pad_idx, tgt_pad_idx,
                 d_model=512, num_layers=6, d_ff=2048,
                 num_heads=8, dropout=0.1, max_len=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_input_dim, d_model, d_ff, num_layers, num_heads, dropout, max_len)
        self.decoder = Decoder(tgt_input_dim, d_model, d_ff, num_layers, num_heads, dropout, max_len)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(device)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len)
        return tgt_mask.to(device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if __name__ == "__main__":
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    tgt = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    tgt_pad_idx = 0
    src_input_dim = 10
    tgt_input_dim = 10
    model = Transformer(src_input_dim, tgt_input_dim, src_pad_idx, tgt_pad_idx).to(device)
    out = model(x, tgt[:, :-1])
    print(out.shape)
