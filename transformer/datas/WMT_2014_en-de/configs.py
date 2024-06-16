# WMT dataset - EN2DE(英语->德语)
WMT_2014_EN2DE_TRAIN = {
    "src": "./train.en",
    "dst": "./train.de",
}

WMT_2014_EN2DE_TEST = {
    "src": "./test.en",
    "dst": "./test.de",
}

WMT_2014_EN2DE_DEV = {
    "src": "./dev.en",
    "dst": "./dev.de",
}

WMT_2014_EN2DE_DICT = {
    "src": "./vocab.50K.en",
    "dst": "./vocab.50K.de",
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

