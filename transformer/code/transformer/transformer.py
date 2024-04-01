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
INPUTMAXSEQLEN = 1000


class Transformer(nn.Module):
    def __init__(
        self,
        dmodel=DMODEL,
        encoder_num=ENCODER_NUM,
        decoder_num=DECODER_NUM,
        headnum=H,
        dff=DFF,
        dictsize_in=DICTSIZE,
        dictsize_out=None,
        input_maxseqlen=INPUTMAXSEQLEN,
        output_maxseqlen=INPUTMAXSEQLEN,
        special_tokens_in=[None, None, None, None],
        special_tokens_out=[None, None, None, None],
    ):
        super(Transformer, self).__init__()

        self.dmodel = dmodel
        self.encoder_num = encoder_num
        self.decoder_num = decoder_num
        self.headnum = headnum
        self.dff = dff
        self.dictsize_src = dictsize_in
        self.dictsize_dst = dictsize_out
        self.input_maxseqlen = input_maxseqlen
        self.output_maxseqlen = output_maxseqlen
        self.padding_idx_src, self.start_idx_src, self.end_idx_src, self.unk_idx_src = (
            special_tokens_in
        )
        self.padding_idx_dst, self.start_idx_dst, self.end_idx_dst, self.unk_idx_dst = (
            special_tokens_out
        )

        # embedding
        self.embedding_src = nn.Embedding(
            self.dictsize_src, self.dmodel, padding_idx=self.padding_idx_src
        )
        self.embedding_dst = self.embedding_src
        # dst and src have their own dicts, need 2 embedding layer
        if self.dictsize_dst is not None:
            assert (
                None not in special_tokens_out
            ), "should specify all dst special tokens like: pad/unk/start/end"

            self.embedding_dst = nn.Embedding(
                self.dictsize_dst, self.dmodel, padding_idx=self.padding_idx_dst
            )
        # this means dst and src use one dict, common for languages with same origin
        # they have much same tokens, this can save a lot spaces
        # origin transformer paper use this way: https://arxiv.org/abs/1706.03762
        else:
            (
                self.padding_idx_dst,
                self.start_idx_dst,
                self.end_idx_dst,
                self.unk_idx_dst,
            ) = special_tokens_in

        self.pos_embedding = self.getPosEncoding(self.input_maxseqlen, self.dmodel)
        self.pos_embedding.requires_grad = False

        # encoder
        self.encoder = TransformerEncoder(
            self.encoder_num, self.dmodel, self.dff, self.headnum
        )

        # decoder
        self.decoder = TransformerDecoder(
            self.decoder_num, self.dmodel, self.dff, self.headnum
        )

        # pre softmax linear
        self.pre_softmax_linear = nn.Linear(self.dmodel, self.dictsize_dst, True)
        # this linear share weight with embedding of dst's weight
        self.pre_softmax_linear.weight = self.embedding_dst.weight

        # softmax
        self.softmax = nn.Softmax(dim=-1)

        # attention mask for train to cover back labels
        self.atten_max_mask = self.getAtteMask(self.output_maxseqlen)
        # atten mask needn't bp
        self.atten_max_mask.requires_grad = False

    def forward(self, tokens, label_tokens=None):
        if self.training:
            assert label_tokens != None, "must provide token for decoder when training"
            return self.forward_train(tokens, label_tokens)
        return self.forward_test(tokens)

    def forward_train(self, src_tokens, decoder_input_tokens):
        tep = self.embedding(src_tokens)
        tep += self.pos_embedding[: src_tokens.shape[-1]]
        encoder_kvs = self.encoder(tep)

        # train need mask
        batch_seqlen = decoder_input_tokens.shape[-1]
        mask = self.atten_max_mask[:batch_seqlen, :batch_seqlen]

        # decoder input
        tep_dec = self.embedding(decoder_input_tokens)
        tep_dec += self.pos_embedding[:batch_seqlen]

        decoder_out = self.decoder(tep_dec, encoder_kvs, mask)

        decoder_out_logit = self.pre_softmax_linear(decoder_out)
        prob = self.softmax(decoder_out_logit)
        return prob

    def forward_test(self, src_tokens):
        tep = self.embedding(src_tokens)
        tep += self.pos_embedding
        encoder_kvs = self.encoder(tep)

        # train need mask
        batch_seqlen = label_tokens.shape[-1]
        mask = self.atten_max_mask[:batch_seqlen, :batch_seqlen]
        decoder_out = self.decoder(self.last_out, encoder_kvs, mask)

        decoder_out_logit = self.pre_softmax_linear(decoder_out)
        prob = self.softmax(decoder_out_logit)
        return prob

    @staticmethod
    def getPosEncoding(input_maxseqlen, dmodel):
        assert dmodel % 2 == 0, "dmodel%2 should be 0"
        poss = pt.arange(input_maxseqlen).unsqueeze(1)  # size=(pos,1)
        dims_half = pt.arange(0, dmodel, 2)  # size = (dmodel//2) 0, 2, 4...
        ret = pt.zeros(input_maxseqlen, dmodel)

        div_num = pt.exp(dims_half / -dmodel * math.log(10000))
        ret[:, 0::2] = pt.sin(poss * div_num)  # size = (pos, dmodel//2)
        ret[:, 1::2] = pt.cos(poss * div_num)
        return ret

    @staticmethod
    def getAtteMask(dims):
        tep = pt.ones([dims, dims], dtype=pt.uint8)
        # Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input,
        # the other elements of the result tensor out are set to 0.
        tril = pt.tril(tep, 0) == 0

        return tril

    @staticmethod
    def getPadMask(batch_lens, input_maxseqlen):
        tep = np.ones([len(batch_lens), input_maxseqlen], dtype=np.uint8)
        for ind, val in enumerate(batch_lens):
            tep[ind, val:] = 0
        return pt.from_numpy(tep == 1)
