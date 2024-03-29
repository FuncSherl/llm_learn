import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import math
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder

# repeat times of encoder and decoder
ENCODER_NUM = 6
DECODER_NUM = 6

DMODEL = 512
H = 8
DFF = 2048

DICTSIZE = 3000
SEQLEN = 1000


class Transformer(nn.Module):
    def __init__(
        self,
        dmodel=DMODEL,
        encoder_num=ENCODER_NUM,
        decoder_num=DECODER_NUM,
        headnum=H,
        dff=DFF,
        dictsize=DICTSIZE,
        seqlen=SEQLEN,
    ):
        super(Transformer, self).__init__()

        self.dmodel = dmodel
        self.encoder_num = encoder_num
        self.decoder_num = decoder_num
        self.headnum = headnum
        self.dff = dff
        self.dictsize = dictsize
        self.seqlen = seqlen

        # embedding
        self.embedding = nn.Embedding(self.dictsize, self.dmodel)
        self.pos_embedding = self.getPosEncoding()

        # encoder
        self.encoder = TransformerEncoder(
            self.encoder_num, self.dmodel, self.dff, self.headnum
        )

        # decoder
        self.decoder = TransformerDecoder(
            self.decoder_num, self.dmodel, self.dff, self.headnum
        )

        # last out
        self.last_out = None

        # pre softmax linear
        self.pre_softmax_linear = nn.Linear(self.dmodel, self.dictsize, True)

        # softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens):
        tep = self.embedding(tokens)
        tep += self.pos_embedding

        encoder_kvs = self.encoder(tep)
        decoder_out = self.decoder(self.last_out, encoder_kvs)
        decoder_out_logit = self.pre_softmax_linear(decoder_out)
        prob = self.softmax(decoder_out_logit)
        return prob

    def getPosEncoding(self):
        assert self.dmodel % 2 == 0, "dmodel%2 should be 0"
        poss = pt.arange(self.seqlen).unsqueeze(1)  # size=(pos,1)
        dims_half = pt.arange(0, self.dmodel, 2)  # size = (dmodel//2) 0, 2, 4...
        ret = pt.zeros(self.seqlen, self.dmodel)

        div_num = pt.exp(dims_half / -self.dmodel * math.log(10000))
        ret[:, 0::2] = pt.sin(poss * div_num)  # size = (pos, dmodel//2)
        ret[:, 1::2] = pt.cos(poss * div_num)
        return ret

    def getAtteMask(self, dims):
        tep = np.ones([dims, dims], dtype=np.uint8)
        tril = np.tril(tep, 0) == 0

        return pt.from_numpy(tril)

    def getPadMask(self, batch_lens):
        tep = np.ones([len(batch_lens), self.seqlen], dtype=np.uint8)
        for ind, val in enumerate(batch_lens):
            tep[ind, val:] = 0
        return pt.from_numpy(tep == 1)
