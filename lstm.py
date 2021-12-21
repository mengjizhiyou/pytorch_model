"""refer to AISUMMER"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        """input gate"""
        self.linear_i_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_i_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_i_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.i_sigmod = nn.Sigmoid()

        """forget gate"""
        self.linear_f_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_f_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_f_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.f_sigmod = nn.Sigmoid()

        """cell memeory"""
        self.linear_c_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_c_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_tanh = nn.Tanh()

        """output gate"""
        self.linear_o_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_o_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_o_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o_sigmod = nn.Sigmoid()

        """hidden memory"""
        self.h_tanh = nn.Tanh()

    def input_gate(self, x, h, c):
        return self.i_sigmod(self.linear_i_x(x) + self.linear_i_h(h) + self.linear_i_c(c))

    def forget_gate(self, x, h, c):
        return self.f_sigmod(self.linear_f_x(x) + self.linear_f_h(h) + self.linear_f_c(c))

    def cell_memory(self, i, f, x, h, c):
        return f * c + i * self.c_tanh(self.linear_c_x(x) + self.linear_c_h(h))

    def output_gate(self, x, h, c_next):
        o = self.o_sigmod(self.linear_o_x(x) + self.linear_o_h(h) + self.linear_o_c(c_next))
        return o * self.h_tanh(c_next)

    def hidden_memory(self, c_next, o):
        return o * self.h_tanh(c_next)

    def init_hidden_cell(self, x):
        """initial hidden and cell"""
        h_0 = x.data.new(x.size(0), self.hidden_dim).zero_()
        c_0 = x.data.new(x.size(0), self.hidden_dim).zero_()
        return (h_0, c_0)

    def forward(self, x, memory=None):
        if memory is not None:
            h, c = memory
        else:
            h, c = self.init_hidden_cell(x)
        i = self.input_gate(x, h, c)  # (x.size(0), hidden_dim)
        f = self.forget_gate(x, h, c)
        c = self.cell_memory(i, f, x, h, c)
        o = self.output_gate(x, h, c)
        h = self.hidden_memory(c, o)
        return o, (h, c)


x = torch.randn(size=(5, 4))
lstm = LSTM(4, 20)
o, (h, u) = lstm(x)
print(o.shape, h.shape, u.shape) 
# torch.Size([5, 20]) torch.Size([5, 20]) torch.Size([5, 20])
