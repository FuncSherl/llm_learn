import torch
from torch.utils.data import Dataset, DataLoader
from .configs import (
    WMT_2014_EN2DE_TRAIN,
    WMT_2014_EN2DE_TEST,
    WMT_2014_EN2DE_DEV,
    WMT_2014_EN_MAX_SEQ_LEN,
    WMT_2014_DE_MAX_SEQ_LEN,
    WMT_2014_EN2DE_DICT,
    SPECIALKEYS,
    UNKSTR,
    STARTSTR,
    ENDSTR,
    PADSTR,
    USE_SAME_DICT,
)
import logging, os
import numpy as np


class Seq2SeqDataset(Dataset):
    def __init__(self, src_data_path, dst_data_path):
        self.src_fp = []
        with open(src_data_path, "r", encoding="utf8") as f:
            self.src_fp = f.readlines()
        self.dst_fp = []
        with open(dst_data_path, "r", encoding="utf8") as f:
            self.dst_fp = f.readlines()

        assert len(self.src_fp) == len(
            self.dst_fp
        ), "dataset src and dst data should have same length"

    def __len__(self):
        return len(self.src_fp)

    def __getitem__(self, index):
        return self.src_fp[index].strip(), self.dst_fp[index].strip()

    def __del__(self):
        pass


# WMT - 2014 - EN2DE dataset
wmt_dataset_train = Seq2SeqDataset(
    WMT_2014_EN2DE_TRAIN["src"], WMT_2014_EN2DE_TRAIN["dst"]
)
wmt_dataset_test = Seq2SeqDataset(
    WMT_2014_EN2DE_TEST["src"], WMT_2014_EN2DE_TEST["dst"]
)
wmt_dataset_dev = Seq2SeqDataset(WMT_2014_EN2DE_DEV["src"], WMT_2014_EN2DE_DEV["dst"])


# dataloaders
def get_train_dataloader(batchsize):
    return DataLoader(
        dataset=wmt_dataset_train,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
    )


def get_test_dataloader(batchsize):
    return DataLoader(
        dataset=wmt_dataset_test,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
    )


def get_dev_dataloader(batchsize):
    return DataLoader(
        dataset=wmt_dataset_dev,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
    )


MAX_SEQLEN_SRC = WMT_2014_EN_MAX_SEQ_LEN
MAX_SEQLEN_DST = WMT_2014_DE_MAX_SEQ_LEN

SRC_DST_DICT_PATH = WMT_2014_EN2DE_DICT


def init_dict():
    logging.info("initializing WMT 2014 src and dst dict... ")
    src_dict_p = SRC_DST_DICT_PATH["src"]
    dst_dict_p = SRC_DST_DICT_PATH["dst"]
    src_dict_set = set(SPECIALKEYS)
    dst_dict_set = set(SPECIALKEYS)
    if os.path.exists(src_dict_p) and os.path.exists(dst_dict_p):
        logging.info("get src and dst str dict files, load dict from files")
        with open(src_dict_p) as f:
            src_dict_set = src_dict_set.union({x.strip() for x in f.readlines()})

        with open(dst_dict_p) as f:
            dst_dict_set = dst_dict_set.union({x.strip() for x in f.readlines()})

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
            "generate dict done, writing to file:\n %s\n%s" % (src_dict_p, dst_dict_p)
        )
        src_dict_set = list(src_dict_set)
        dst_dict_set = list(dst_dict_set)
        with open(src_dict_p, "w", encoding="utf8") as f:
            f.writelines(src_dict_set)
        with open(dst_dict_p, "w", encoding="utf8") as f:
            f.writelines(dst_dict_set)

    logging.info(
        "initialize WMT 2014 dict done: \n%s size: %d\n%s size: %d\n"
        % (src_dict_p, len(src_dict_set), dst_dict_p, len(dst_dict_set))
    )
    return src_dict_set, dst_dict_set


SRC_DICT_SET, DST_DICT_SET = init_dict()

# use same dict for en and de, this is special for this job, for other languages maybe different
if USE_SAME_DICT:
    SRC_DICT_SET = SRC_DICT_SET | DST_DICT_SET
    DST_DICT_SET = SRC_DICT_SET

SRC_DICT_SET = list(SRC_DICT_SET)
DST_DICT_SET = list(DST_DICT_SET)


def init_word2token(dictlist):
    ret = {}
    for i in range(len(dictlist)):
        ret[dictlist[i]] = i
    return ret


def init_token2word(dictlist):
    ret = {}
    for i in range(len(dictlist)):
        ret[i] = dictlist[i]
    return ret


SRC_WORD2TOKEN = init_word2token(SRC_DICT_SET)
DST_WORD2TOKEN = init_word2token(DST_DICT_SET)

SRC_TOKEN2WORD = init_token2word(SRC_DICT_SET)
DST_TOKEN2WORD = init_token2word(DST_DICT_SET)


def batch_word2token(bsword, wtdict, addstart=True, addend=True):
    maxlen = 0
    for i in bsword:
        maxlen = max(maxlen, len(i))
    ret = np.full(
        [len(bsword), maxlen + int(addstart) + int(addend)],
        fill_value=wtdict[PADSTR],
        dtype=np.int32,
    )
    if addstart:
        ret[:, 0] = wtdict[STARTSTR]

    ed_id = wtdict[ENDSTR]
    st_ind = int(addstart)
    for indi, i in enumerate(bsword):
        for indj, j in enumerate(i):
            if j in wtdict:
                ret[indi][indj + st_ind] = wtdict[j]
            else:
                ret[indi][indj + st_ind] = wtdict[UNKSTR]
        if addend:
            ret[indi][len(i) + st_ind] = ed_id
    return ret


def batch_token2word(bstoken, wtdict):
    ret = []
    for indi, i in enumerate(bstoken):
        kep = []
        for indj, j in enumerate(i):
            if j in wtdict:
                kep.append(wtdict[j])
            else:
                kep.append(UNKSTR)
        ret.append(kep)
    return ret


if __name__ == "__main__":
    cnt = 0
    for d, l in get_train_dataloader(4):
        cnt += len(d)
        # print (d)
        # print (l)
        for i in range(len(d)):
            # print (d[i], " --> ", l[i])
            pass
        # break
    print("train total cnt : ", cnt)

    cnt = 0
    for d, l in get_test_dataloader(5):
        cnt += len(d)
        for i in range(len(d)):
            # print (d[i], " --> ", l[i])
            pass
        # break
    print("test total cnt : ", cnt)
