import os
import os.path as op

# WMT dataset - EN2DE(英语->德语)
WMT_2014_EN2DE_TRAIN = {
    "src": op.join(os.path.dirname(__file__), "./train.en"),
    "dst": op.join(os.path.dirname(__file__), "./train.de"),
}

WMT_2014_EN2DE_TEST = {
    "src": op.join(os.path.dirname(__file__), "./test.en"),
    "dst": op.join(os.path.dirname(__file__), "./test.de"),
}

WMT_2014_EN2DE_DEV = {
    "src": op.join(os.path.dirname(__file__), "./dev.en"),
    "dst": op.join(os.path.dirname(__file__), "./dev.de"),
}

WMT_2014_EN2DE_DICT = {
    "src": op.join(os.path.dirname(__file__), "./vocab.50K.en"),
    "dst": op.join(os.path.dirname(__file__), "./vocab.50K.de"),
}
"""
使用shell统计数据得到,作为固定参数, 这里取一个略大的值:
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ./train.en ./test.en ./dev.en
"""
WMT_2014_EN_MAX_SEQ_LEN = 150
"""
这里取一个略大的值
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ./train.de ./test.de ./dev.de
"""
WMT_2014_DE_MAX_SEQ_LEN = 150

# use same dict for en and de, this is special for this job, for other languages maybe different
USE_SAME_DICT = True

SPECIALKEYS = ["<unk>", "<s>", "</s>", "<pad>"]
UNKSTR = SPECIALKEYS[0]
STARTSTR = SPECIALKEYS[1]
ENDSTR = SPECIALKEYS[2]
PADSTR = SPECIALKEYS[3]
