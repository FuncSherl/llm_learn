#!/bin/bash

# referto : https://awslabs.github.io/sockeye/tutorials/wmt_large.html
wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en'
wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de'
for YEAR in 2012 2013 2014; do
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.en"
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.de"
done
cat newstest{2012,2013}.en >dev.en
cat newstest{2012,2013}.de >dev.de
cp newstest2014.en test.en
cp newstest2014.de test.de