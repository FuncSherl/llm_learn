from transformer import decoder, encoder, transformer
from dataloader import wmt_train_dataloader, wmt_test_dataloader, wmt_dev_dataloader
from configs import (
    WMT_2014_EN2DE_DICT,
    BATCHSIZE,
    SPECIALKEYS,
    PADSTR,
    DMODEL,
    ENCODER_NUM,
    DECODER_NUM,
    HEADNUM,
    DFF,
    WMT_2014_EN_MAX_SEQ_LEN,
    WMT_2014_DE_MAX_SEQ_LEN,
    UNKSTR,
    STARTSTR,
    ENDSTR,
)
import os, logging

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


class WMT2014EN2DE:
    def __init__(self, use_same_dict=True) -> None:
        self.train_dataloader = wmt_train_dataloader
        self.test_dataloader = wmt_test_dataloader
        self.dev_dataloader = wmt_dev_dataloader

        self.src_dict_set, self.dst_dict_set = self.init_dict()

        # use same dict for en and de, this is special for this job, for other languages maybe different
        if use_same_dict:
            self.src_dict_set = self.src_dict_set | self.dst_dict_set
            self.dst_dict_set = self.src_dict_set

        self.src_word2token = self.init_word2token(self.src_dict_set)
        self.dst_word2token = self.init_word2token(self.dst_dict_set)

        self.src_token2word = self.init_token2word(self.src_dict_set)
        self.dst_token2word = self.init_token2word(self.dst_dict_set)
        self.src_special_tokens = [
            self.src_word2token[PADSTR],
            self.src_word2token[STARTSTR],
            self.src_word2token[ENDSTR],
            self.src_word2token[UNKSTR],
        ]
        self.dst_special_tokens = [
            self.dst_word2token[PADSTR],
            self.dst_word2token[STARTSTR],
            self.dst_word2token[ENDSTR],
            self.dst_word2token[UNKSTR],
        ]

        self.transformer_model = transformer.Transformer(
            DMODEL,
            ENCODER_NUM,
            DECODER_NUM,
            HEADNUM,
            DFF,
            len(self.src_dict_set),
            None if use_same_dict else len(self.dst_dict_set),
            WMT_2014_EN_MAX_SEQ_LEN,
            WMT_2014_DE_MAX_SEQ_LEN,
            self.src_special_tokens,
            self.dst_special_tokens,
        )

    def init_dict(self):
        logging.info("initializing src and dst dict... ")
        src_dict_p = WMT_2014_EN2DE_DICT["src"]
        dst_dict_p = WMT_2014_EN2DE_DICT["dst"]
        src_dict_set = set(SPECIALKEYS)
        dst_dict_set = set(SPECIALKEYS)
        if os.path.exists(src_dict_p) and os.path.exists(dst_dict_p):
            logging.info("get src and dst str dict files, load dict from files")
            with open(src_dict_p) as f:
                src_dict_set = list(
                    src_dict_set.union({x.strip() for x in f.readlines()})
                )
            with open(dst_dict_p) as f:
                dst_dict_set = list(
                    dst_dict_set.union({x.strip() for x in f.readlines()})
                )
        else:
            raise LookupError(
                "can't find one or more follow dict files: \n%s\n%s"
                % (src_dict_p, dst_dict_p)
            )
            """
            重新生成dict的耗时非常久,这里推荐使用shell:
                awk  '{for (i=1;i<=NF;++i) a[$i]=1;} END{for (j in a) print j;}'  ../datas/WMT_2014_en-de/train.en
            但是话说回来,并不是所有单词都需要放到dict里面,比如： "a.m.-10.30"  就没有意义,
            因此这里可能使用数据下载处提供的../datas/WMT_2014_en-de/vocab.50k.en更好
            实际使用shell跑出来所有train数据的词典约816971个单词,在训练过程中大多单词项没用,
            因此这里直接报错, 不推荐重新本地生成dict
            """
            logging.info("regenerating dict from train/test/dev datasets...")
            for dataloader in [
                wmt_train_dataloader,
                wmt_test_dataloader,
                wmt_dev_dataloader,
            ]:
                cnt = 0
                for srcd, dstd in dataloader:  # one batch data
                    cnt += len(srcd)
                    for srcstr, dststr in zip(srcd, dstd):  # every single data, one str
                        src_dict_set = src_dict_set.union(set(srcstr.split()))
                        dst_dict_set = dst_dict_set.union(set(dststr.split()))
                    logging.info(
                        "%d/%d  src dict size: %d, dst dict size: %d"
                        % (
                            cnt,
                            len(dataloader.dataset),
                            len(src_dict_set),
                            len(dst_dict_set),
                        )
                    )
            logging.info(
                "generate dict done, writing to file:\n %s\n%s"
                % (src_dict_p, dst_dict_p)
            )
            src_dict_set = list(src_dict_set)
            dst_dict_set = list(dst_dict_set)
            with open(src_dict_p, "w", encoding="utf8") as f:
                f.writelines(src_dict_set)
            with open(dst_dict_p, "w", encoding="utf8") as f:
                f.writelines(dst_dict_set)

        logging.info(
            "initialize dict done: \n%s size: %d\n%s size: %d\n"
            % (src_dict_p, len(src_dict_set), dst_dict_p, len(dst_dict_set))
        )
        return src_dict_set, dst_dict_set

    def init_word2token(self, dictlist):
        ret = {}
        for i in range(len(dictlist)):
            ret[dictlist[i]] = i
        return ret

    def init_token2word(self, dictlist):
        ret = {}
        for i in range(len(dictlist)):
            ret[i] = dictlist[i]
        return ret

    def train(self):
        self.transformer_model.train()

    def test(self):
        self.transformer_model.eval()


if __name__ == "__main__":
    wmtproc = WMT2014EN2DE()
