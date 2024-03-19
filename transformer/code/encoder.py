import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import *


class transformer_encoder():
    def __init__(self, repeat_num=6, dmodel=512, dff = 1024, headnum=8) -> None:
        self.repeat_num = repeat_num
        self.dmodel = dmodel
        self.headnum = headnum
        self.dff = dff

    def forward(self, x):
        ret = []
        tep = x
        for i in range(self.repeat_num):
            # sublayer 1
            kep = tep
            tep = multi_head_attention(
                tep,
                tep,
                tep,
                self.dmodel / self.headnum,
                self.dmodel / self.headnum,
                self.headnum,
            )
            tep = nn.LayerNorm([self.dmodel])(tep + kep)
            
            # sublayer 2
            kep = tep
            tep = feed_forward(tep, self.dff)
            tep = nn.LayerNorm([self.dmodel])(tep + kep)
            ret.append(tep)
        return ret
