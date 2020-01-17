import numpy as np
from torch import nn


class SGDRScheduler(nn.Module):
    global_epoch = 0
    all_epoch = 0
    cur_drop_prob = 0.
    def __init__(self, dropblock):
        super(SGDRScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = 0.

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        #self.dropblock.drop_prob = np.abs((0 + 0.5 * 0.1 * (1 + np.cos(np.pi * SGDRScheduler.global_epoch / SGDRScheduler.all_epoch)))-0.1)
        #SGDRScheduler.cur_drop_prob = self.dropblock.drop_prob
        ix = np.log2(self.global_epoch / 10 + 1).astype(np.int)
        T_cur = self.global_epoch - 10 * (2 ** (ix) - 1)
        T_i = (10 * 2 ** ix)
        self.dropblock.drop_prob = np.abs((0 + 0.5 * 0.1 * (1 + np.cos(np.pi * T_cur / T_i)))-0.1)
        SGDRScheduler.cur_drop_prob = self.dropblock.drop_prob

class LinearScheduler(nn.Module):
    global_epoch = 0
    num_epochs = 0
    def __init__(self, dropblock, start_value=0., stop_value=0.1):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=self.num_epochs)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
            self.dropblock.drop_prob = self.drop_values[self.global_epoch]
