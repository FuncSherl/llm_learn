import os, logging
from .configs import DATASET_NAME, DATASET_SUB_NAMES
from datasets import (
    load_dataset_builder,
    load_dataset,
    get_dataset_split_names,
    get_dataset_config_names,
)

cache_dir = os.path.dirname(__file__)

logging.info("%s: Getting all sub datasets names..." % (DATASET_NAME))
DATASET_SUB_NAMES = get_dataset_config_names(DATASET_NAME)

logging.info("%s: All sub datasets is: %s" % (DATASET_NAME, str(DATASET_SUB_NAMES)))

example_subdat = DATASET_SUB_NAMES[0]
SPLITS = get_dataset_split_names(DATASET_NAME, example_subdat)
logging.info("%s: Split of %s is %s" % (DATASET_NAME, example_subdat, str(SPLITS)))


# dataloaders ['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en']
def get_train_dataloader(batchsize, language_pair="de-en", *args, **kwargs):
    assert language_pair in DATASET_SUB_NAMES, "language_pair should in %s, get: %s" % (
        str(DATASET_SUB_NAMES),
        language_pair,
    )
    dataset = load_dataset(
        DATASET_NAME,
        language_pair,
        split="train",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )


def get_test_dataloader(batchsize, *args, **kwargs):
    pass


def get_dev_dataloader(batchsize, *args, **kwargs):
    pass


if __name__ == "__main__":
    pass
