import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import *


class TransformerDecoder(nn.Module):
    def __init__(self, repeat_num=6, dmodel=512, dff=1024, headnum=8) -> None:
        super(TransformerDecoder, self).__init__()
        self.repeat_num = repeat_num
        self.dmodel = dmodel
        self.headnum = headnum
        self.dff = dff

        self.multhead_att_fir = [
            MultiHeadAttention(
                self.dmodel / self.headnum,
                self.dmodel / self.headnum,
                self.dmodel,
                self.headnum,
            )
            for i in range(self.repeat_num)
        ]

        self.multhead_att_sec = [
            MultiHeadAttention(
                self.dmodel / self.headnum,
                self.dmodel / self.headnum,
                self.dmodel,
                self.headnum,
            )
            for i in range(self.repeat_num)
        ]

        self.feed_forward = [FeedForward(dmodel, dff) for i in range(self.repeat_num)]

    def forward(self, x, encoder_kv, mask = None):
        ret = []
        tep = x
        for i in range(self.repeat_num):
            # sublayer 1
            kep = tep
            # train need mask
            tep = self.multhead_att_fir[i](tep, tep, tep, mask)
            tep = nn.LayerNorm([self.dmodel])(tep + kep)

            # sublayer 2
            kep = tep
            # no need mask
            tep = self.multhead_att_sec[i](tep, encoder_kv, encoder_kv)
            tep = nn.LayerNorm([self.dmodel])(tep + kep)

            # sublayer 3
            kep = tep
            tep = self.feed_forward(tep, self.dff)
            tep = nn.LayerNorm([self.dmodel])(tep + kep)
            ret.append(tep)
        return ret
