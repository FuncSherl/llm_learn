import torch
from torch.utils.data import Dataset, DataLoader
from .configs import (
    IWSLT_15_EN2VI_TRAIN,
    IWSLT_15_EN2VI_TEST,
    IWSLT_15_EN2VI_DEV,
    IWSLT_15_EN_MAX_SEQ_LEN,
    IWSLT_15_VI_MAX_SEQ_LEN,
)


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
iwslt_dataset_train = Seq2SeqDataset(
    IWSLT_15_EN2VI_TRAIN["src"], IWSLT_15_EN2VI_TRAIN["dst"]
)
iwslt_dataset_test = Seq2SeqDataset(
    IWSLT_15_EN2VI_TEST["src"], IWSLT_15_EN2VI_TEST["dst"]
)
iwslt_dataset_dev = Seq2SeqDataset(IWSLT_15_EN2VI_DEV["src"], IWSLT_15_EN2VI_DEV["dst"])


# dataloaders
def get_train_dataloader(batchsize):
    return DataLoader(
        dataset=iwslt_dataset_train,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
    )


def get_test_dataloader(batchsize):
    return DataLoader(
        dataset=iwslt_dataset_test,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
    )


def get_dev_dataloader(batchsize):
    return DataLoader(
        dataset=iwslt_dataset_dev,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
    )


MAX_SEQLEN_SRC = IWSLT_15_EN_MAX_SEQ_LEN
MAX_SEQLEN_DST = IWSLT_15_VI_MAX_SEQ_LEN

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
    for d, l in get_test_dataloader(4):
        cnt += len(d)
        for i in range(len(d)):
            # print (d[i], " --> ", l[i])
            pass
        # break
    print("test total cnt : ", cnt)
