import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_dim * 10, embedding_dim, padding_idx=0)
        self.embedding.weight.data = nn.Parameter(torch.empty(input_dim, embedding_dim))

        # positive direction
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)

        # negative direction
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim)

        self.linear = nn.Linear(2 * hidden_dim, output_dim)

    def forward_states(self, x):
        mask = TriangularCausalMask(self.input_dim, direction='positive')._mask
        xx = torch.einsum('nhs,nse->nhe', mask, x)  # (1, input_dim, embedding_dim)
        for i in range(self.input_dim):
            if i == 0:
                out, (h, c) = self.lstm1(xx[:, i].unsqueeze(0))
                fw_out = out  # (1, 1, hidden_dim)
            else:
                out, (h, c) = self.lstm1(xx[:, i].unsqueeze(0), (h, c))
                fw_out = torch.concat((fw_out, out), dim=1)
        return fw_out

    def backward_states(self, x):
        mask = TriangularCausalMask(self.input_dim, direction='negative')._mask
        xx = torch.einsum('nhs,nse->nhe', mask, x)  # (1, input_dim, embedding_dim)
        for i in range(self.input_dim):
            if i == 0:
                out, (h, c) = self.lstm2(xx[:, i].unsqueeze(0))
                bw_out = out
            else:
                out, (h, c) = self.lstm2(xx[:, i].unsqueeze(0), (h, c))
                bw_out = torch.concat((bw_out, out), dim=1)
        return bw_out

    def forward(self, x):
        x = self.embedding(x.long())
        fw_out = self.forward_states(x)
        bw_out = self.backward_states(x)
        for i in range(self.input_dim):
            """前向的output/hidden与后向的output/hidden连接，注意时间"""
            o1 = torch.index_select(fw_out, 1, torch.tensor([i]))
            o2 = torch.index_select(bw_out, 1, torch.tensor([self.input_dim - i - 1]))
            if i == 0:
                output = self.linear(torch.concat((o1, o2), dim=-1))  # concat, mean, sum
            else:
                out = self.linear(torch.concat((o1, o2), dim=-1))  # concat, mean, sum
                output = torch.concat((output, out), dim=-1)
        return output.squeeze(1)


class TriangularCausalMask:
    """
    构建两个掩码：
    1. 正方向LSTM：掩码行号与时间一致
    2. 反方向LSTM：掩码行号与时间的逆序一致
    """

    def __init__(self, seq_len, direction):
        mask_shape = [1, seq_len, seq_len]
        meet_index = seq_len // 2  # 当negative 和 positive 的t=meet_index时，相遇

        with torch.no_grad():
            if direction == 'positive':
                self._mask = torch.tril(torch.ones(mask_shape))
            else:
                self._mask = torch.zeros(mask_shape)  # t=T---->t=1

                """当正向LSTM相遇时，t=0:meet_index的信息已经获取"""
                mask_shape2 = [1, meet_index, meet_index]  # for negative
                self._mask[:, meet_index:, :meet_index] = torch.ones(mask_shape2)

                """当正向LSTM相遇时，t=meet_index:的信息实时获取"""
                mask_shape3 = [1, meet_index - 1, meet_index - 1]  # for negative
                self._mask[:, meet_index + 1:, meet_index:-1] = torch.tril(torch.ones(mask_shape3))

    @property
    def mask(self):
        return self._mask


x = torch.randint(1, 8, size=(1, 8)).float()
seq_len = input_dim = x.size(1)  # input_dim=seq_len
embedding_dim = 10

model = BiLSTM(input_dim, embedding_dim=10, hidden_dim=64, output_dim=1)
output = model(x)
print(output.size())  # torch.Size([1, 8])
