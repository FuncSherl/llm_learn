import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import math


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

def feed_forward(x, dff):
    hidden = nn.Linear(x.shape[-1], dff, bias=True)(x)
    relu = nn.ReLU()(hidden)
    return nn.Linear(dff, x.shape[-1], bias=True)(relu)

def pos_encoding(pos, dmodel):
    assert dmodel%2 == 0, "dmodel%2 should be 0"
    poss = pt.arange(pos).unsqueeze(1)  #size=(pos,1)
    dims_half = pt.arange(0, dmodel, 2)  # size = (dmodel//2) 0, 2, 4...
    ret = pt.zeros(pos, dmodel)    
    
    div_num = pt.exp(dims_half/-dmodel*math.log(10000))
    ret[:, 0::2] = pt.sin(poss*div_num) # size = (pos, dmodel//2)
    ret[:, 1::2] = pt.cos(poss*div_num)
    return ret
    
    
