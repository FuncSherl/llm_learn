from transformer import decoder, encoder, transformer
from dataloader import wmt_train_dataloader, wmt_test_dataloader, wmt_dev_dataloader
from configs import WMT_2014_EN2DE_DICT, BATCHSIZE, SPECIALKEYS
import os, logging

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


class WMT2014EN2DE:
    def __init__(self) -> None:
        self.train_dataloader = wmt_train_dataloader
        self.test_dataloader = wmt_test_dataloader
        self.dev_dataloader = wmt_dev_dataloader

        self.src_dict_set, self.dst_dict_set = self.init_dict()

    def init_dict(self):
        logging.info("initializing src and dst dict... ")
        src_dict_p = WMT_2014_EN2DE_DICT["src"]
        dst_dict_p = WMT_2014_EN2DE_DICT["dst"]
        src_dict_set = set(SPECIALKEYS)
        dst_dict_set = set(SPECIALKEYS)
        if os.path.exists(src_dict_p) and os.path.exists(dst_dict_p):
            logging.info("get src and dst str dict files, load dict from files")
            with open("src_dict_p") as f:
                src_dict_set = [x.strip() for x in f.readlines()]
            with open("dst_dict_p") as f:
                dst_dict_set = [x.strip() for x in f.readlines()]
        else:
            logging.info("regenerating dict from train/test/dev datasets...")
            for dataloader in [
                wmt_train_dataloader,
                wmt_test_dataloader,
                wmt_dev_dataloader,
            ]:
                for srcd, dstd in dataloader:  # one batch data
                    for srcstr, dststr in zip(srcd, dstd):  # every single data, one str
                        src_dict_set = src_dict_set.union(set(srcstr.split()))
                        dst_dict_set = dst_dict_set.union(set(dststr.split()))
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

    def train(self):
        pass

    def test(self):
        pass


if __name__ == "__main__":
    wmtproc = WMT2014EN2DE()
