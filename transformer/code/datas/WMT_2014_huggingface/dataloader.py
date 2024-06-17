import os, logging
from .configs import DATASET_NAME, DATASET_SUB_NAMES
from torch.utils.data import DataLoader
from datasets import (
    load_dataset_builder,
    load_dataset,
    get_dataset_split_names,
    get_dataset_config_names,
)

cache_dir = os.path.join(os.path.dirname(__file__), "__datacache__")

logging.info("%s: Getting all sub datasets names..." % (DATASET_NAME))
DATASET_SUB_NAMES = get_dataset_config_names(DATASET_NAME)

logging.info("%s: All sub datasets is: %s" % (DATASET_NAME, str(DATASET_SUB_NAMES)))

example_subdat = DATASET_SUB_NAMES[0]
SPLITS = get_dataset_split_names(DATASET_NAME, example_subdat)
logging.info("%s: Split of %s is %s" % (DATASET_NAME, example_subdat, str(SPLITS)))


def load_hf_dataset(bs, subsetname, split):
    assert subsetname in DATASET_SUB_NAMES, "language_pair should in %s, get: %s" % (
        str(DATASET_SUB_NAMES),
        subsetname,
    )
    return load_dataset(
        DATASET_NAME,
        subsetname,
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )


# dataloaders ['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en']
def get_train_dataloader(
    batchsize, language_src="en", language_dst="de", *args, **kwargs
):
    subsetname = language_src + "-" + language_dst
    if subsetname not in DATASET_SUB_NAMES:
        subsetname = language_dst + "-" + language_src
    dataset = load_hf_dataset(batchsize, subsetname, "train")

    return DataLoader(
        dataset=dataset,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
        collate_fn=lambda dat: (
            [d["translation"][language_src] for d in dat],
            [d["translation"][language_dst] for d in dat],
        ),
    )


def get_test_dataloader(
    batchsize, language_src="en", language_dst="de", *args, **kwargs
):
    subsetname = language_src + "-" + language_dst
    if subsetname not in DATASET_SUB_NAMES:
        subsetname = language_dst + "-" + language_src
    dataset = load_hf_dataset(batchsize, subsetname, "test")

    return DataLoader(
        dataset=dataset,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
        collate_fn=lambda dat: (
            [d["translation"][language_src] for d in dat],
            [d["translation"][language_dst] for d in dat],
        ),
    )


def get_dev_dataloader(
    batchsize, language_src="en", language_dst="de", *args, **kwargs
):
    subsetname = language_src + "-" + language_dst
    if subsetname not in DATASET_SUB_NAMES:
        subsetname = language_dst + "-" + language_src
    dataset = load_hf_dataset(batchsize, subsetname, "validation")

    return DataLoader(
        dataset=dataset,  # 传入的数据集, 必须参数
        batch_size=batchsize,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱
        num_workers=2,  # 进程数, 0表示只有主进程
        collate_fn=lambda dat: (
            [d["translation"][language_src] for d in dat],
            [d["translation"][language_dst] for d in dat],
        ),
    )


if __name__ == "__main__":
    dat = get_train_dataloader(5)
    cnt = 0
    for i in dat:
        if cnt > 5:
            break
        print(i)
        cnt += 1
