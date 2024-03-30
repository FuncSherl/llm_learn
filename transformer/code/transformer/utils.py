import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dk, dv, dmodel, headnum):
        super(MultiHeadAttention, self).__init__()
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.headnum = headnum
        # first linear layers
        self.q_linears = [
            nn.Linear(self.dmodel, self.dk, bias=False) for x in range(self.headnum)
        ]
        self.k_linears = [
            nn.Linear(self.dmodel, self.dk, bias=False) for x in range(self.headnum)
        ]
        self.v_linears = [
            nn.Linear(self.dmodel, self.dv, bias=False) for x in range(self.headnum)
        ]
        # last linear
        self.last_linear = nn.Linear(
            self.dk * self.headnum, self.dk * self.headnum, bias=False
        )

    @staticmethod
    def dot_product_attention(q, k, v, mask = None, mask_val = -1e9):
        tep = pt.matmul(q, k.t())
        dk_sqr = pt.sqrt(k.shape[-1])
        tep = tep / dk_sqr
        if mask:
            tep = tep.masked_fill(mask, mask_val)
        tep = F.softmax(tep)
        return pt.matmul(tep, v)

    def forward(self, q, k, v, mask = None):
        qall = [f(q) for f in self.q_linears]
        kall = [f(k) for f in self.k_linears]
        vall = [f(v) for f in self.v_linears]
        # dot product
        headall = [
            self.dot_product_attention(qall[x], kall[x], vall[x], mask)
            for x in range(self.headnum)
        ]
        head_cat = pt.cat(headall, dim=-1)
        return self.last_linear(head_cat)


class FeedForward(nn.Module):
    def __init__(self, dinput, dff):
        super(FeedForward, self).__init__()
        self.dinput = dinput
        self.dff = dff
        self.hidden = nn.Linear(self.dinput, self.dff, bias=True)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.dff, self.dinput, bias=True)
        
    def forward(self, x):
        tep = self.hidden(x)
        tep = self.relu(tep)
        return self.out(tep)


# def multi_head_attention(q, k, v, dk, dv, headnum):
#     qall = [nn.Linear(q.shape[-1], dk, bias=False)(q) for x in range(headnum)]
#     kall = [nn.Linear(k.shape[-1], dk, bias=False)(k) for x in range(headnum)]
#     vall = [nn.Linear(v.shape[-1], dv, bias=False)(v) for x in range(headnum)]

#     headall = [dot_product_attention(qall[x], kall[x], vall[x]) for x in range(headnum)]
#     head_cat = pt.cat(headall, dim=-1)
#     return nn.Linear(dk*headnum, dk*headnum, bias=False)(head_cat)


# def feed_forward(x, dff):
#     hidden = nn.Linear(x.shape[-1], dff, bias=True)(x)
#     relu = nn.ReLU()(hidden)
#     return nn.Linear(dff, x.shape[-1], bias=True)(relu)
