#!/bin/bash

# referto : https://awslabs.github.io/sockeye/tutorials/wmt_large.html
wget 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en'
wget 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi'
for YEAR in 2012 2013; do
    wget "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst${YEAR}.en"
    wget "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst${YEAR}.vi"
done
cat tst2012.en >dev.en
cat tst2012.vi >dev.vi
cp tst2013.en test.en
cp tst2013.vi test.vi

# other datas, not necessary
wget "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.en"
wget "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.vi"

wget "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/dict.en-vi"