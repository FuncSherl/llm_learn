import os
import os.path as op

# IWSLT dataset - EN2VI(英语->越南语)
IWSLT_15_EN2VI_TRAIN = {
    "src": op.join(os.path.dirname(__file__), "./train.en"),
    "dst": op.join(os.path.dirname(__file__), "./train.de"),
}
IWSLT_15_EN2VI_TEST = {
    "src": op.join(os.path.dirname(__file__), "./test.en"),
    "dst": op.join(os.path.dirname(__file__), "./test.de"),
}
IWSLT_15_EN2VI_DEV = {
    "src": op.join(os.path.dirname(__file__), "./dev.en"),
    "dst": op.join(os.path.dirname(__file__), "./dev.de"),
}
IWSLT_15_EN2VI_DICT = {
    "src": op.join(os.path.dirname(__file__), "./vocab.en"),
    "dst": op.join(os.path.dirname(__file__), "./vocab.de"),
}
"""
使用shell统计数据得到,作为固定参数, 这里取一个略大的值:
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ./train.en ./test.en ./dev.en

"""
IWSLT_15_EN_MAX_SEQ_LEN = 650
"""
这里取一个略大的值
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ./train.vi ./test.vi ./dev.vi
"""
IWSLT_15_VI_MAX_SEQ_LEN = 860

# use same dict for en and de, this is special for this job, for other languages maybe different
USE_SAME_DICT = True

SPECIALKEYS = ["<unk>", "<s>", "</s>", "<pad>"]
UNKSTR = SPECIALKEYS[0]
STARTSTR = SPECIALKEYS[1]
ENDSTR = SPECIALKEYS[2]
PADSTR = SPECIALKEYS[3]
