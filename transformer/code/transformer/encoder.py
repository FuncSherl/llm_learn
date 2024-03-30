import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import *


class TransformerEncoder(nn.Module):
    def __init__(self, repeat_num=6, dmodel=512, dff=1024, headnum=8) -> None:
        super(TransformerEncoder, self).__init__()
        self.repeat_num = repeat_num
        self.dmodel = dmodel
        self.headnum = headnum
        self.dff = dff

        self.multhead_att = [
            MultiHeadAttention(
                self.dmodel / self.headnum,
                self.dmodel / self.headnum,
                self.dmodel,
                self.headnum,
            )
            for i in range(self.repeat_num)
        ]

        self.feed_forward = [FeedForward(dmodel, dff) for i in range(self.repeat_num)]

    def forward(self, x):        
        tep = x
        for i in range(self.repeat_num):
            # sublayer 1
            kep = tep
            tep = self.multhead_att[i](tep, tep, tep)
            tep = nn.LayerNorm([self.dmodel])(tep + kep)

            # sublayer 2
            kep = tep
            tep = self.feed_forward[i](tep)
            tep = nn.LayerNorm([self.dmodel])(tep + kep)
        return tep
