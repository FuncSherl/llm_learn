import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import math, logging
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder

# repeat times of encoder and decoder
ENCODER_NUM = 6
DECODER_NUM = 6

DMODEL = 512
H = 8
DFF = 2048

DICTSIZE = 50000
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
        special_tokens_in=None,
        special_tokens_out=None,
        dropout_prob=0.1,
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

        self.padding_idx_src, self.start_idx_src, self.end_idx_src = (
            special_tokens_in[0],
            special_tokens_in[1],
            special_tokens_in[2],
        )
        self.padding_idx_dst, self.start_idx_dst, self.end_idx_dst = (
            special_tokens_out[0],
            special_tokens_out[1],
            special_tokens_out[2],
        )
        self.dropout_prob = dropout_prob

        # embedding
        self.embedding_src = nn.Embedding(
            self.dictsize_src, self.dmodel, padding_idx=self.padding_idx_src
        )
        self.embedding_dst = self.embedding_src
        # dst and src have their own dicts, need 2 embedding layer
        if self.dictsize_dst is not None:
            assert (
                self.padding_idx_dst is not None
            ), "should specify padding dst special token id"

            self.embedding_dst = nn.Embedding(
                self.dictsize_dst, self.dmodel, padding_idx=self.padding_idx_dst
            )
        # this means dst and src use one dict, common for languages with same origin
        # they have much same tokens, this can save a lot spaces
        # origin transformer paper use this way: https://arxiv.org/abs/1706.03762
        else:
            self.padding_idx_dst = self.padding_idx_src
            self.start_idx_dst = self.start_idx_src
            self.dictsize_dst = self.dictsize_src

        self.pos_embedding = self.getPosEncoding(self.input_maxseqlen, self.dmodel)
        self.pos_embedding.requires_grad = False

        # encoder
        self.encoder = TransformerEncoder(
            self.encoder_num, self.dmodel, self.dff, self.headnum, self.dropout_prob
        )

        # decoder
        self.decoder = TransformerDecoder(
            self.decoder_num, self.dmodel, self.dff, self.headnum, self.dropout_prob
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
        device = src_tokens.device
        tep = self.embedding_src(src_tokens)

        tep += self.pos_embedding[None, : src_tokens.shape[-1]].to(device)
        encoder_kvs = self.encoder(tep)

        # train need mask
        batch_seqlen = decoder_input_tokens.shape[-1]
        mask = self.atten_max_mask[None, :batch_seqlen, :batch_seqlen].to(device)

        # decoder input
        tep_dec = self.embedding_dst(decoder_input_tokens)
        tep_dec += self.pos_embedding[None, :batch_seqlen].to(device)

        decoder_out = self.decoder(tep_dec, encoder_kvs, mask)

        decoder_out_logit = self.pre_softmax_linear(decoder_out)
        # prob = self.softmax(decoder_out_logit)
        return decoder_out_logit

    """
        input one or more token, out one token probs
    """

    def forward_test(self, src_tokens):
        device = src_tokens.device
        tep = self.embedding_src(src_tokens)
        tep += self.pos_embedding[None, : src_tokens.shape[-1]].to(device)
        encoder_kvs = self.encoder(tep)

        batch_s = src_tokens.shape[0]  # bs x seqlen x dmodel
        # mask = self.atten_max_mask[:batch_seqlen, :batch_seqlen]
        outputs_tokens = np.full([batch_s, 1], self.start_idx_dst, dtype=np.int32)
        logging.info("encoder done, get kvs: " + str(encoder_kvs.shape))

        cnt = 0
        end_flags = np.full([batch_s], False, dtype=np.bool_)
        while np.any(~end_flags):
            logging.info("test running %d decoder iter..." % (cnt))
            logging.info("get output size: " + str(outputs_tokens.shape))
            tensor_out = pt.tensor(outputs_tokens, dtype=pt.int32).to(device)
            dst_emb = self.embedding_dst(tensor_out)
            dst_emb += self.pos_embedding[None, : tensor_out.shape[-1]].to(device)
            decoder_out = self.decoder(dst_emb, encoder_kvs)

            decoder_out_logit = self.pre_softmax_linear(decoder_out)
            probs = self.softmax(decoder_out_logit)[:, -1:]
            ntoken = pt.argmax(probs, -1).cpu().numpy()
            outputs_tokens = np.append(outputs_tokens, ntoken, axis=-1)
            end_flags = end_flags | (ntoken == self.end_idx_dst)
            logging.info("test running %d decoder iter done\n" % (cnt))
            cnt += 1
            if outputs_tokens.shape[-1] >= self.output_maxseqlen:
                logging.info("test get maxseqlen, finishing...")
                break

        return outputs_tokens

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
