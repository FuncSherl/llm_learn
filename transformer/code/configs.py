# WMT dataset - EN2DE(英语->德语)
WMT_2014_EN2DE_TRAIN = {
    "src": "../datas/WMT_2014_en-de/train.en",
    "dst": "../datas/WMT_2014_en-de/train.de",
}

WMT_2014_EN2DE_TEST = {
    "src": "../datas/WMT_2014_en-de/test.en",
    "dst": "../datas/WMT_2014_en-de/test.de",
}

WMT_2014_EN2DE_DEV = {
    "src": "../datas/WMT_2014_en-de/dev.en",
    "dst": "../datas/WMT_2014_en-de/dev.de",
}

WMT_2014_EN2DE_DICT = {
    "src": "../datas/WMT_2014_en-de/vocab.50K.en",
    "dst": "../datas/WMT_2014_en-de/vocab.50K.de",
}
"""
使用shell统计数据得到,作为固定参数:
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ../datas/WMT_2014_en-de/train.en ../datas/WMT_2014_en-de/test.en ../datas/WMT_2014_en-de/dev.en
"""
WMT_2014_EN_MAX_SEQ_LEN = 150
"""
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ../datas/WMT_2014_en-de/train.de ../datas/WMT_2014_en-de/test.de ../datas/WMT_2014_en-de/dev.de
"""
WMT_2014_DE_MAX_SEQ_LEN = 150

# IWSLT dataset - EN2VI(英语->越南语)
IWSLT_15_EN2VI_TRAIN = {
    "src": "../datas/IWSLT_15_en-vi/train.en",
    "dst": "../datas/IWSLT_15_en-vi/train.de",
}
IWSLT_15_EN2VI_TEST = {
    "src": "../datas/IWSLT_15_en-vi/test.en",
    "dst": "../datas/IWSLT_15_en-vi/test.de",
}
IWSLT_15_EN2VI_DEV = {
    "src": "../datas/IWSLT_15_en-vi/dev.en",
    "dst": "../datas/IWSLT_15_en-vi/dev.de",
}
IWSLT_15_EN2VI_DICT = {
    "src": "../datas/IWSLT_15_en-vi/vocab.50K.en",
    "dst": "../datas/IWSLT_15_en-vi/vocab.50K.de",
}
"""
使用shell统计数据得到,作为固定参数:
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ../datas/IWSLT_15_en-vi/train.en ../datas/IWSLT_15_en-vi/test.en ../datas/IWSLT_15_en-vi/dev.en

"""
IWSLT_15_EN_MAX_SEQ_LEN = 650
"""
awk  'BEGIN{kep=0;} {if(NF>kep){kep=NF; print $0; print NF;}; } END{print kep;}'  \
    ../datas/IWSLT_15_en-vi/train.vi ../datas/IWSLT_15_en-vi/test.vi ../datas/IWSLT_15_en-vi/dev.vi
"""
IWSLT_15_VI_MAX_SEQ_LEN = 860

# train configs
EPOCHS=4
BATCHSIZE = 10
SPECIALKEYS = ["<unk>", "<s>", "</s>", "<pad>"]
UNKSTR = SPECIALKEYS[0]
STARTSTR = SPECIALKEYS[1]
ENDSTR = SPECIALKEYS[2]
PADSTR = SPECIALKEYS[3]

# model configs
ENCODER_NUM = 6
DECODER_NUM = 6

DMODEL = 512
HEADNUM = 8
DFF = 2048

WARMUP_STEPS=4000
