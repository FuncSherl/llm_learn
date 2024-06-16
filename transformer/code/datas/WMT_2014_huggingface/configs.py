import os
import os.path as op

SPECIALKEYS = ["<unk>", "<s>", "</s>", "<pad>"]
UNKSTR = SPECIALKEYS[0]
STARTSTR = SPECIALKEYS[1]
ENDSTR = SPECIALKEYS[2]
PADSTR = SPECIALKEYS[3]


# huggingface configs
DATASET_NAME = "wmt14"
DATASET_SUB_NAMES = ['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en']