import torch as pt
import torch.nn as nn
import torch.nn.functional as F


def dot_product_attention(q, k, v):
    tep = pt.matmul(q, k.t())
    dk_sqr = pt.sqrt(k.shape[-1])

    tep = tep / dk_sqr
    tep = F.softmax(tep)
    return pt.matmul(tep, v)


def multi_head_attention(q, k, v, dk, dv, headnum):
    qall = [nn.Linear(q.shape[-1], dk, bias=False)(q) for x in range(headnum)]
    kall = [nn.Linear(k.shape[-1], dk, bias=False)(k) for x in range(headnum)]
    vall = [nn.Linear(v.shape[-1], dv, bias=False)(v) for x in range(headnum)]

    headall = [dot_product_attention(qall[x], kall[x], vall[x]) for x in range(headnum)]
    head_cat = pt.cat(headall, dim=-1)
    return nn.Linear(dk*headnum, dk*headnum, bias=False)(head_cat)
