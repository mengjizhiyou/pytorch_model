from __future__ import print_function
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_class=None):
        super(MLP, self).__init__()
        self.num_class = num_class
        self.m = nn.Sequential(nn.Linear(input_size, hidden_size),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_size, num_class))
        self.reset_params()

    def reset_params(self):
        # 初始化可学习的参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None, reg=0):
        pred = self.m(x)
        print(pred)
        if self.num_class is None:
            if y is not None:
                y = Variable(y)
                mse = F.mse_loss(pred, y)
                mae = F.l1_loss(pred, y)
                return pred, mae, mse
            else:
                return pred

        if self.num_class > 1:
            logits = F.log_softmax(pred, dim=1)
            if y is not None:
                y = Variable(y)
                loss = F.nll_loss(logits, y) + ((1 / 2) * 0.8 * reg)
                pred = logits.data.max(1, keepdim=True)[1]
                debug_tmp = pred.eq(y.data.view_as(pred)).sum().float()
                y_size = float(y.size()[0])
                acc = debug_tmp / y_size
                return logits, loss, acc
            else:
                return logits
