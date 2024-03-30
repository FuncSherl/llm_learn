import torch
from torch.utils.data import Dataset, DataLoader
from configs import WMT_2014_EN2DE_TRAIN, WMT_2014_EN2DE_TEST, WMT_2014_EN2DE_DEV


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
        return self.src_fp[index], self.dst_fp[index]

    def __del__(self):
        if not self.src_fp.closed:
            self.src_fp.close()
        if not self.dst_fp.closed:
            self.dst_fp.close()
