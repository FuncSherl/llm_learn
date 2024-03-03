import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

# repeat times of encoder and decoder
ENCODER_NUM = 6
DECODER_NUM = 6

DMODEL = 512
H=8
DFF=2048
