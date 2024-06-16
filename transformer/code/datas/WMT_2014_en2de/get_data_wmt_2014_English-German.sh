#!/bin/bash

# referto : https://awslabs.github.io/sockeye/tutorials/wmt_large.html
wget -nc  'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en'
wget -nc  'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de'
for YEAR in 2012 2013 2014; do
    wget -nc  "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.en"
    wget -nc  "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.de"
done
cat newstest{2012,2013}.en >dev.en
cat newstest{2012,2013}.de >dev.de
cp newstest2014.en test.en
cp newstest2014.de test.de

# other datas, not necessary
wget -nc  "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en"
wget -nc  "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de"

wget -nc  "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de"