#!/bin/bash
## scrap websites to get font zips 

# create main library's tarball
# run dataflow on entire datasets and store it in different folder

pipenv run python scripts/preprocessing/py/apply-supervised-filtering.py \
  --input-folder=data/preprocessed/train \
  --input-folder=data/preprocessed/val \
  --input-folder=data/preprocessed/test \
  --charset=lowercase \
  --output-folder=data/filtered/lowercase \
  --filter-model=models/supervised-model-1-lowercase/model \
  --batch-size=500

pipenv run python scripts/preprocessing/py/apply-supervised-filtering.py \
  --input-folder=data/preprocessed/train \
  --input-folder=data/preprocessed/val \
  --input-folder=data/preprocessed/test \
  --charset=uppercase \
  --output-folder=data/filtered/uppercase \
  --filter-model=models/supervised-model-1-uppercase/model \
  --batch-size=500

pipenv run python scripts/preprocessing/py/apply-supervised-filtering.py \
  --input-folder=data/preprocessed/train \
  --input-folder=data/preprocessed/val \
  --input-folder=data/preprocessed/test \
  --charset=numbers \
  --output-folder=data/filtered/numbers \
  --filter-model=models/supervised-model-1-numbers/model \
  --batch-size=500
  
