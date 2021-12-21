import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        """reset gate"""
        self.re_x = nn.Linear(input_dim, hidden_dim)
        self.re_h = nn.Linear(hidden_dim, hidden_dim)
        self.re_sigmoid = nn.Sigmoid()
        self.re_tanh = nn.Tanh()

        """update gate"""
        self.up_x = nn.Linear(input_dim, hidden_dim)
        self.up_h = nn.Linear(hidden_dim, hidden_dim)
        self.up_sigmoid = nn.Sigmoid()

        """hidden memory"""
        self.h_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.h_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_tanh = nn.Tanh()

    def reset_gate(self, x, h):
        re = self.re_sigmoid(self.re_x(x) + self.re_h(h))
        r = self.re_tanh(re * self.h_h(h) + self.h_x(x))
        return r

    def update_gate(self, x, h):
        up = self.up_sigmoid(self.up_x(x) + self.up_h(h))
        u = up * h
        return up, u

    def hidden_memory(self, r, up,u):
        h = r * (1 - up) + u
        return h

    def output(self, h):
        return h

    def init_hidden(self, x):
        h_0 = x.data.new(x.size(0), self.hidden_dim).zero_()
        return h_0

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x)
        r = self.reset_gate(x, h)
        up, u = self.update_gate(x, h)
        h = self.hidden_memory(r, up, u)
        o = self.output(h)
        return o, h


x = torch.randn(size=(5, 4))
gru = GRU(4, 20)
o, h = gru(x)
print(o.shape, h.shape)
# torch.Size([5, 20]) torch.Size([5, 20])
