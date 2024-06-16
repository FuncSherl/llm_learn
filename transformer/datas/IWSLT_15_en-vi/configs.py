# IWSLT dataset - EN2VI(英语->越南语)
IWSLT_15_EN2VI_TRAIN = {
    "src": "./train.en",
    "dst": "./train.de",
}
IWSLT_15_EN2VI_TEST = {
    "src": "./test.en",
    "dst": "./test.de",
}
IWSLT_15_EN2VI_DEV = {
    "src": "./dev.en",
    "dst": "./dev.de",
}
IWSLT_15_EN2VI_DICT = {
    "src": "./vocab.en",
    "dst": "./vocab.de",
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

