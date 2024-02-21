import torch as pt
import torch.nn as nn
import torch.nn.functional as F

def dot_product_attention(q, k, v):
    tep = pt.matmul(q, k.t())
    dk_sqr = pt.sqrt(k.shape[-1])
    
    tep = tep / dk_sqr
    tep = F.softmax(tep)
    return pt.matmul(tep, v)