import torch
from torch.utils.data import Dataset, DataLoader
from .configs import (
    WMT_2014_EN2DE_TRAIN,
    WMT_2014_EN2DE_TEST,
    WMT_2014_EN2DE_DEV,
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
